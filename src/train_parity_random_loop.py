import os
import sys
import uuid
from random import randint

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
from generate_training_data import generate_prompt_matrix_parity
from test_func import test_model, test_model_adaptive


def _extract_int_flag(argv, name, default):
    flag = f"--{name}"
    if flag not in argv:
        return default
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        raise ValueError(f"Missing value for {flag}")
    value = int(argv[idx + 1])
    del argv[idx:idx + 2]
    return value


def train(model, args, loop_min, loop_max):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.training.train_steps - args.training.curriculum.points.end * args.training.curriculum.points.interval,
        eta_min=0.0,
    )
    curriculum = Curriculum(args.training.curriculum)
    if args.training.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999, use_num_updates=False)
    starting_step = 0
    bsize = args.training.batch_size
    pbar = tqdm(range(starting_step, args.training.train_steps))
    loss_func = nn.CrossEntropyLoss()
    best_acc = float("-inf")
    for i in pbar:
        xs, batch_num, ys, mask = generate_prompt_matrix_parity(
            bsize,
            min_num_digits=1,
            max_num_digits=curriculum.n_points,
            max_len=curriculum.n_points + 1,
        )
        xs = torch.tensor(convert_to_one_hot(xs))
        xs = xs.cuda()
        ys = ys.cuda()
        mask = mask.cuda()

        with torch.enable_grad():
            optimizer.zero_grad()
            horizon = loop_max
            states = model.looped_forward(xs, horizon=horizon)
            loop_k = randint(loop_min, loop_max)
            outputs = states[loop_k - 1]
            loss = loss_func(outputs[mask == 1], ys[mask == 1])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if i > (args.training.curriculum.points.end * args.training.curriculum.points.interval):
                scheduler.step()
                if args.training.ema:
                    if i == (args.training.curriculum.points.end * args.training.curriculum.points.interval) + 1:
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
                    "loop_k": loop_k,
                },
                step=i,
            )

        if (i) % 1000 == 0:
            print("current max training length = ", curriculum.n_points - 1)
            test_acc_current = test_model(
                model,
                curriculum.n_points - 1,
                512,
                generate_prompt_matrix_parity,
                convert_to_one_hot,
                one_hot_to_int,
                exact_match_accuracy,
            )
            test_acc_chosen_current, _ = test_model_adaptive(
                model,
                curriculum.n_points - 1,
                512,
                generate_prompt_matrix_parity,
                convert_to_one_hot,
                one_hot_to_int,
                exact_match_accuracy,
            )
            print("test_acc_current = ", test_acc_current)
            print("test_acc_chosen_current = ", test_acc_chosen_current)
            print("index", _)
            test_len = args.training.test_len
            print("test_len = ", test_len)
            test_acc = test_model(
                model,
                test_len,
                512,
                generate_prompt_matrix_parity,
                convert_to_one_hot,
                one_hot_to_int,
                exact_match_accuracy,
            )
            test_acc_chosen, _ = test_model_adaptive(
                model,
                test_len,
                512,
                generate_prompt_matrix_parity,
                convert_to_one_hot,
                one_hot_to_int,
                exact_match_accuracy,
            )
            print("test_acc = ", test_acc)
            print("test_acc_chosen = ", test_acc_chosen)
            print("index", _)
            wandb.log(
                {
                    "test_acc": test_acc,
                    "test_acc_chosen": test_acc_chosen,
                },
                step=i,
            )
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(model, os.path.join(args.out_dir, "best.pt"))
        curriculum.update()
        pbar.set_description(f"loss {loss}")

    # test after training
    test_len = args.training.test_len
    if args.training.ema:
        with ema.average_parameters():
            test_acc_final = test_model(
                model,
                test_len,
                6400,
                generate_prompt_matrix_parity,
                convert_to_one_hot,
                one_hot_to_int,
                exact_match_accuracy,
            )
        test_acc_chosen_final, _ = test_model_adaptive(
            model,
            test_len,
            6400,
            generate_prompt_matrix_parity,
            convert_to_one_hot,
            one_hot_to_int,
            exact_match_accuracy,
        )
        print("test_acc_final = ", test_acc_final)
        print("test_acc_chosen_final = ", test_acc_chosen_final)
    else:
        test_acc_final = test_model(
            model,
            test_len,
            6400,
            generate_prompt_matrix_parity,
            convert_to_one_hot,
            one_hot_to_int,
            exact_match_accuracy,
        )
        test_acc_chosen_final, _ = test_model_adaptive(
            model,
            test_len,
            6400,
            generate_prompt_matrix_parity,
            convert_to_one_hot,
            one_hot_to_int,
            exact_match_accuracy,
        )
        print("test_acc_final = ", test_acc_final)
        print("test_acc_chosen_final = ", test_acc_chosen_final)
    if test_acc_final >= best_acc:
        best_acc = test_acc_final
        torch.save(model, os.path.join(args.out_dir, "best.pt"))
    wandb.log(
        {
            "test_acc_final": test_acc_final,
            "test_acc_chosen_final": test_acc_chosen_final,
        },
        step=i,
    )
    torch.save(model, os.path.join(args.out_dir, f"model.pt"))
    if args.training.ema:
        with ema.average_parameters():
            torch.save(model, os.path.join(args.out_dir, f"model_ema.pt"))


def main(args, loop_min, loop_max):
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
    train(model, args, loop_min, loop_max)


if __name__ == "__main__":
    loop_min = _extract_int_flag(sys.argv, "loop_min", 1)
    loop_max = _extract_int_flag(sys.argv, "loop_max", 10)
    if loop_min < 1 or loop_max < loop_min:
        raise ValueError("Invalid loop range: require 1 <= loop_min <= loop_max")

    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family == "gpt2"
    assert args.training.task == "parity"
    print(f"Running with: {args}")
    print(f"Random loop range: [{loop_min}, {loop_max}]")

    if not args.test_run:
        run_id = f"{uuid.uuid4()}_randloop_{loop_min}_{loop_max}"

        out_dir = args.out_dir
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

    main(args, loop_min, loop_max)
