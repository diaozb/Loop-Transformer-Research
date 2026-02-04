import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import torch.nn as nn
from curriculum import Curriculum
from schema import schema
from models import build_general_model
import wandb
from utils import convert_to_one_hot, one_hot_to_int, exact_match_accuracy
from torch_ema import ExponentialMovingAverage
from generate_training_data import generate_prompt_matrix_parity, generate_prompt_matrix_copy, generate_prompt_matrix_addition, generate_prompt_matrix_multi, generate_prompt_matrix_sum_reverse, generate_prompt_matrix_dict, generate_prompt_matrix_mod_add, generate_prompt_matrix_mod_add_digits
from test_func import test_model, test_model_adaptive, test_model_multi, test_model_multi_adaptive, test_model_dict, test_model_dict_adaptive

generate_function_map = {
    "parity": generate_prompt_matrix_parity,
    "copy": generate_prompt_matrix_copy,
    "addition": generate_prompt_matrix_addition,
    "multi": generate_prompt_matrix_multi,
    "sum_reverse": generate_prompt_matrix_sum_reverse,
    "dict": generate_prompt_matrix_dict,
    "mod_add": generate_prompt_matrix_mod_add,
    "mod_add_digits": generate_prompt_matrix_mod_add_digits
    }

test_function_map = {
    "parity": test_model,
    "copy": test_model,
    "addition": test_model,
    "multi": test_model_multi,
    "sum_reverse": test_model,
    "dict": test_model_dict,
    "mod_add": test_model,
    "mod_add_digits": test_model
    }

test_function_map_adaptive = {
    "parity": test_model_adaptive,
    "copy": test_model_adaptive,
    "addition": test_model_adaptive,
    "multi": test_model_multi_adaptive,
    "sum_reverse": test_model_adaptive,
    "dict": test_model_dict_adaptive,
    "mod_add": test_model_adaptive,
    "mod_add_digits": test_model_adaptive
    }

def train(model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.training.train_steps-args.training.curriculum.points.end*args.training.curriculum.points.interval, eta_min=0.0)
    curriculum = Curriculum(args.training.curriculum)
    if args.training.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999, use_num_updates=False)
    starting_step = 0
    bsize = args.training.batch_size
    pbar = tqdm(range(starting_step, args.training.train_steps))
    loss_func = nn.CrossEntropyLoss()
    best_acc = float("-inf")
    for i in pbar:   
        if args.training.task != "multi":
            # multiplication task needs two lengths
            if args.training.task == "mod_add":
                xs, batch_num, ys, mask = generate_prompt_matrix_mod_add(
                    bsize,
                    min_num_digits=1,
                    max_num_digits=curriculum.n_points,
                    max_len=curriculum.n_points + 1,
                    modulus=args.training.modulus,
                )
            elif args.training.task == "mod_add_digits":
                xs, batch_num, ys, mask = generate_prompt_matrix_mod_add_digits(
                    bsize,
                    min_num_digits=1,
                    max_num_digits=curriculum.n_points,
                    max_len=3 * (curriculum.n_points + 1),
                    modulus=args.training.modulus,
                )
            else:
                xs, batch_num, ys, mask = generate_function_map[args.training.task](bsize, min_num_digits = 1, max_num_digits = curriculum.n_points, max_len = curriculum.n_points+1)
        else:
            xs, batch_num, batch_num_1, ys, mask = generate_prompt_matrix_multi(bsize, min_num_digits = 1, max_num_digits = curriculum.n_points, max_len = curriculum.n_points+1)
        #  since max_num_digits is unincliusive, the actual max number of digits is curriculum.n_points - 1
        if args.training.task != "dict":
            if args.training.task in ("mod_add", "mod_add_digits"):
                xs = torch.tensor(convert_to_one_hot(xs, n_dims=args.model.n_dims))
            else:
                xs = torch.tensor(convert_to_one_hot(xs))
        xs = xs.cuda()
        ys = ys.cuda()
        
        with torch.enable_grad():
            optimizer.zero_grad()
            if args.training.task != "multi":
                # multiplication task needs to be trained with a larger horizon
                states = model.looped_forward(xs, horizon = curriculum.n_points+2)
            else:
                states = model.looped_forward(xs, horizon = curriculum.n_points*2)

            states_list = []
            for t in range(bsize):
                if args.training.task != "multi":
                    states_list.append(states[batch_num[t].item()-1][t])
                else: 
                    states_list.append(states[batch_num[t].item()*batch_num_1[t].item()-1][t])
            
            outputs = torch.stack(states_list)
            loss = loss_func(outputs[mask==1], ys[mask==1])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if i>(args.training.curriculum.points.end*args.training.curriculum.points.interval):
                scheduler.step()
                if args.training.ema:
                    if i==(args.training.curriculum.points.end*args.training.curriculum.points.interval)+1:
                        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999, use_num_updates=False)
                    else:
                        ema.update()

        if (i) % args.wandb.log_every_steps == 0:
            print(f"Step {i}, loss {loss}")
            wandb.log(
                {
                    "training_loss": loss,
                    "gradient_norm": grad_norm, 
                    "n_points": curriculum.n_points,
                },
                step=i,
            )
            
        if (i) % 1000 == 0:
            print("current max training length = ", curriculum.n_points-1)
            converter = convert_to_one_hot
            if args.training.task in ("mod_add", "mod_add_digits"):
                converter = lambda arr: convert_to_one_hot(arr, n_dims=args.model.n_dims)
            test_generate = generate_function_map[args.training.task]
            if args.training.task == "mod_add":
                test_generate = lambda *a, **k: generate_prompt_matrix_mod_add(*a, **k, modulus=args.training.modulus)
            elif args.training.task == "mod_add_digits":
                def _test_generate_mod_add_digits(b, max_len, min_num_digits, max_num_digits):
                    result_len = len(str(args.training.modulus - 1))
                    L = max_num_digits - 1
                    needed = 2 * L + 2 + result_len
                    return generate_prompt_matrix_mod_add_digits(
                        b,
                        max_len=needed,
                        min_num_digits=min_num_digits,
                        max_num_digits=max_num_digits,
                        modulus=args.training.modulus,
                    )
                test_generate = _test_generate_mod_add_digits
            test_acc_current = test_function_map[args.training.task](model, curriculum.n_points-1, 512, test_generate, converter, one_hot_to_int, exact_match_accuracy)
            test_acc_chosen_current, _ = test_function_map_adaptive[args.training.task](model, curriculum.n_points-1, 512, test_generate, converter, one_hot_to_int, exact_match_accuracy)
            print("test_acc_current = ", test_acc_current)
            print("test_acc_chosen_current = ", test_acc_chosen_current)
            print("index", _)
            test_len = args.training.test_len
            print("test_len = ", test_len)
            test_acc = test_function_map[args.training.task](model, test_len, 512, test_generate, converter, one_hot_to_int, exact_match_accuracy)
            test_acc_chosen, _ = test_function_map_adaptive[args.training.task](model, test_len, 512, test_generate, converter, one_hot_to_int, exact_match_accuracy)
            print("test_acc = ", test_acc)
            print("test_acc_chosen = ", test_acc_chosen)
            print("index", _)
            wandb.log(
                {
                    "test_acc": test_acc,
                    "test_acc_chosen": test_acc_chosen
                },
                step=i,
            )
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(model, os.path.join(args.out_dir, "best.pt"))
        curriculum.update()
        pbar.set_description(f"loss {loss}")

    # test after training
    converter = convert_to_one_hot
    if args.training.task in ("mod_add", "mod_add_digits"):
        converter = lambda arr: convert_to_one_hot(arr, n_dims=args.model.n_dims)
    test_generate = generate_function_map[args.training.task]
    if args.training.task == "mod_add":
        test_generate = lambda *a, **k: generate_prompt_matrix_mod_add(*a, **k, modulus=args.training.modulus)
    elif args.training.task == "mod_add_digits":
        def _test_generate_mod_add_digits(b, max_len, min_num_digits, max_num_digits):
            result_len = len(str(args.training.modulus - 1))
            L = max_num_digits - 1
            needed = 2 * L + 2 + result_len
            return generate_prompt_matrix_mod_add_digits(
                b,
                max_len=needed,
                min_num_digits=min_num_digits,
                max_num_digits=max_num_digits,
                modulus=args.training.modulus,
            )
        test_generate = _test_generate_mod_add_digits

    if args.training.ema:
        with ema.average_parameters():
            test_acc_final = test_function_map[args.training.task](model, test_len, 6400, test_generate, converter, one_hot_to_int, exact_match_accuracy)
        test_acc_chosen_final, _ = test_function_map_adaptive[args.training.task](model, test_len, 6400, test_generate, converter, one_hot_to_int, exact_match_accuracy)
        print("test_acc_final = ", test_acc_final)
        print("test_acc_chosen_final = ", test_acc_chosen_final)
    else:       
        test_acc_final = test_function_map[args.training.task](model, test_len, 6400, test_generate, converter, one_hot_to_int, exact_match_accuracy)
        test_acc_chosen_final, _ = test_function_map_adaptive[args.training.task](model, test_len, 6400, test_generate, converter, one_hot_to_int, exact_match_accuracy)
        print("test_acc_final = ", test_acc_final)
        print("test_acc_chosen_final = ", test_acc_chosen_final)
    if test_acc_final >= best_acc:
        best_acc = test_acc_final
        torch.save(model, os.path.join(args.out_dir, "best.pt"))
    wandb.log(
        {
            "test_acc_final": test_acc_final,
            "test_acc_chosen_final": test_acc_chosen_final
        },
        step=i,
    )
    torch.save(model, os.path.join(args.out_dir, f"model.pt"))
    if args.training.ema:
        with ema.average_parameters():
            torch.save(model, os.path.join(args.out_dir, f"model_ema.pt"))

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        args.training.train_steps = 1
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_general_model(args.model)
    model = model.to(torch.float32)
    model.cuda()
    model.train()
    train(model, args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family == "gpt2"
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = str(uuid.uuid4())

        out_dir = args.out_dir
        if args.training.task in ("mod_add", "mod_add_digits"):
            out_dir = os.path.join(out_dir, f"mod_{args.training.modulus}")
        if args.model.use_wpe:
            out_dir = os.path.join(out_dir, "wpe")
        if args.model.use_rope:
            out_dir = os.path.join(out_dir, "rope")
        out_dir = os.path.join(out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
