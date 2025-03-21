from model import train_model
import argparse
import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_path = os.path.join(project_root, 'data.yaml')
    
    parser = argparse.ArgumentParser(description='Train YOLOv11s model for brain tumor detection')
    parser.add_argument('--data', type=str, default=default_data_path, help='path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='input image size')
    parser.add_argument('--resume', action='store_true', help='resume training from last checkpoint')
    parser.add_argument('--resume-path', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--run-id', type=str, default=None, help='MLflow run ID to continue logging')
    args = parser.parse_args()

    # Train model
    results, run_id = train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch_size=args.batch_size,
        resume=args.resume,
        resume_path=args.resume_path,
        run_id=args.run_id
    )
    
    print(f"Training completed. Results: {results}")
    print(f"MLflow run ID: {run_id}")

if __name__ == '__main__':
    main()