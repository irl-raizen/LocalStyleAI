#!/usr/bin/env python3
"""
LocalStyleAI — Main Entry Point

Usage:
    python run.py --mode train    --style anime_clean
    python run.py --mode generate --prompt "a sunset over the ocean" --style ghibli_clean
    python run.py --mode prepare
    python run.py --mode server
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="LocalStyleAI",
        description="Local Style AI Generator — train LoRAs and generate styled images.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "generate", "prepare", "server"],
        help="Operation mode: 'train', 'generate', 'prepare' (dataset), or 'server'.",
    )

    # Generation args
    parser.add_argument("--prompt", type=str, default="a fantasy landscape",
                        help="Text prompt for generation (used with --mode generate).")
    parser.add_argument("--style", type=str, default="default",
                        help="Style to apply: anime_clean, ghibli_clean, lineart_clean, or default.")
    parser.add_argument("--output", type=str, default="exports/output.png",
                        help="Output file path for generated image.")

    # Training args
    parser.add_argument("--steps", type=int, default=1200, help="Training steps.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution.")

    # Server args
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")

    args = parser.parse_args()

    if args.mode == "train":
        from src.train.train_lora import train
        # Build a namespace that train() expects
        train_args = argparse.Namespace(
            style=args.style,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            resolution=args.resolution,
        )
        train(train_args)

    elif args.mode == "generate":
        from src.inference.generate import generate_image
        generate_image(
            prompt=args.prompt,
            style=args.style,
            output_path=args.output,
        )
        print(f"Image saved to: {args.output}")

    elif args.mode == "prepare":
        from src.data.dataset_prep import prepare_dataset
        prepare_dataset(resolution=args.resolution)

    elif args.mode == "server":
        import uvicorn
        print(f"Starting LocalStyleAI server on {args.host}:{args.port}")
        uvicorn.run("api.app:app", host=args.host, port=args.port, reload=True)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
