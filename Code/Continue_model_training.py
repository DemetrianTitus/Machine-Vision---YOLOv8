from ultralytics import YOLO
import multiprocessing
import argparse
import logging
import torch

def parse_args():
    # Argument parsing section: Handles command-line arguments for model configuration, dataset configuration, training parameters, and other settings
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument("--model", type=str, default="yolov8s.yaml", help="Model configuration file")
    parser.add_argument("--checkpoint", type=str, default="runs/detect/train/weights/last.pt", help="Path to checkpoint file") # Added checkpoint
    parser.add_argument("--data", type=str, default="config.yaml", help="Dataset configuration file")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training")
    parser.add_argument("--save_dir", type=str, default="runs/detect/train", help="Directory to save training results")
    parser.add_argument("--save_period", type=int, default=50, help="Frequency of saving checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for training")
    args = parser.parse_args()
    return args

def setup_logging():
    # Logging setup section: Configures logging to display informational messages with timestamps
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger

def main():
    # Main function section: Parses arguments, sets up logging, loads the model, and starts the training process
    args = parse_args()
    logger = setup_logging()

    logger.info("Loading model checkpoint")
    # Load the YOLO model from the specified checkpoint and move it to the specified device
    model = YOLO(args.checkpoint).to(args.device)

    # Training section: Trains the YOLO model with the specified parameters and mixed precision enabled
    model.train(data=args.data, epochs=args.epochs, device=args.device, save_dir=args.save_dir,
                save_period=args.save_period, batch=args.batch_size, imgsz=args.img_size, amp=True)
    
    logger.info("Training completed")

if __name__ == '__main__':
    # Multiprocessing support section: Ensures the script can be safely run with multiprocessing
    multiprocessing.freeze_support()
    main()
