import os
import argparse
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import seaborn as sns

def track_metrics(days=7, output_dir="metrics_reports"):
    """
    Theo dõi và phân tích metrics từ MLflow
    
    Args:
        days (int): Số ngày dữ liệu cần phân tích
        output_dir (str): Thư mục để lưu báo cáo
    """
    # Thiết lập MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.tracking.MlflowClient()
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy thời gian bắt đầu (số ngày trước)
    start_time = datetime.now() - timedelta(days=days)
    
    # Lấy tất cả các experiment
    experiments = client.search_experiments()
    
    all_metrics = []
    
    for experiment in experiments:
        print(f"Đang phân tích experiment: {experiment.name}")
        
        # Lấy tất cả các run trong experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attribute.start_time > {int(start_time.timestamp() * 1000)}"
        )
        
        for run in runs:
            run_data = {
                "experiment_name": experiment.name,
                "run_id": run.info.run_id,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0),
                "status": run.info.status
            }
            
            # Thêm tất cả các tham số
            for key, value in run.data.params.items():
                run_data[f"param_{key}"] = value
            
            # Thêm tất cả các metrics
            for key, value in run.data.metrics.items():
                run_data[f"metric_{key}"] = value
            
            all_metrics.append(run_data)
    
    if not all_metrics:
        print("Không tìm thấy dữ liệu metrics nào trong khoảng thời gian đã chọn.")
        return
    
    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame(all_metrics)
    
    # Lưu dữ liệu thô
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"metrics_data_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Tạo báo cáo tổng quan
    report = {
        "generated_at": datetime.now().isoformat(),
        "period_days": days,
        "total_runs": len(df),
        "experiments": df["experiment_name"].nunique(),
        "successful_runs": len(df[df["status"] == "FINISHED"]),
        "failed_runs": len(df[df["status"] != "FINISHED"])
    }
    
    # Thêm thống kê về metrics
    metrics_columns = [col for col in df.columns if col.startswith("metric_")]
    for col in metrics_columns:
        metric_name = col.replace("metric_", "")
        if df[col].dtype in [int, float]:
            report[f"{metric_name}_avg"] = df[col].mean()
            report[f"{metric_name}_min"] = df[col].min()
            report[f"{metric_name}_max"] = df[col].max()
    
    # Lưu báo cáo tổng quan
    report_path = os.path.join(output_dir, f"report_summary_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    
    # Tạo các biểu đồ trực quan
    create_visualizations(df, output_dir, timestamp)
    
    print(f"Báo cáo đã được lưu tại: {output_dir}")
    print(f"Dữ liệu thô: {csv_path}")
    print(f"Báo cáo tổng quan: {report_path}")
    
    return df, report

def create_visualizations(df, output_dir, timestamp):
    """
    Tạo các biểu đồ trực quan từ dữ liệu metrics
    
    Args:
        df (DataFrame): DataFrame chứa dữ liệu metrics
        output_dir (str): Thư mục để lưu biểu đồ
        timestamp (str): Timestamp để đặt tên file
    """
    # Thư mục cho biểu đồ
    viz_dir = os.path.join(output_dir, f"visualizations_{timestamp}")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Thiết lập style cho biểu đồ
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # 1. Biểu đồ số lượng run theo experiment
    plt.figure(figsize=(12, 6))
    experiment_counts = df['experiment_name'].value_counts()
    experiment_counts.plot(kind='bar')
    plt.title('Số lượng Run theo Experiment')
    plt.xlabel('Experiment')
    plt.ylabel('Số lượng Run')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'runs_by_experiment.png'))
    plt.close()
    
    # 2. Biểu đồ tỷ lệ run thành công/thất bại
    plt.figure(figsize=(8, 8))
    status_counts = df['status'].value_counts()
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Tỷ lệ Run thành công/thất bại')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'run_status_pie.png'))
    plt.close()
    
    # 3. Biểu đồ timeline các run
    plt.figure(figsize=(14, 6))
    df_sorted = df.sort_values('start_time')
    for i, experiment in enumerate(df['experiment_name'].unique()):
        exp_df = df_sorted[df_sorted['experiment_name'] == experiment]
        plt.scatter(exp_df['start_time'], [i] * len(exp_df), label=experiment, s=50)
    
    plt.yticks(range(len(df['experiment_name'].unique())), df['experiment_name'].unique())
    plt.title('Timeline của các Run')
    plt.xlabel('Thời gian')
    plt.ylabel('Experiment')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'run_timeline.png'))
    plt.close()
    
    # 4. Biểu đồ metrics theo thời gian
    metrics_columns = [col for col in df.columns if col.startswith("metric_")]
    
    for col in metrics_columns:
        if df[col].dtype in [int, float]:
            metric_name = col.replace("metric_", "")
            plt.figure(figsize=(12, 6))
            
            for experiment in df['experiment_name'].unique():
                exp_df = df[df['experiment_name'] == experiment].sort_values('start_time')
                if not exp_df.empty and col in exp_df.columns:
                    plt.plot(exp_df['start_time'], exp_df[col], 'o-', label=experiment)
            
            plt.title(f'{metric_name} theo thời gian')
            plt.xlabel('Thời gian')
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'metric_{metric_name}_timeline.png'))
            plt.close()
    
    # 5. Biểu đồ phân phối các metrics
    for col in metrics_columns:
        if df[col].dtype in [int, float]:
            metric_name = col.replace("metric_", "")
            plt.figure(figsize=(10, 6))
            
            sns.histplot(df[col], kde=True)
            plt.title(f'Phân phối của {metric_name}')
            plt.xlabel(metric_name)
            plt.ylabel('Tần suất')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'metric_{metric_name}_distribution.png'))
            plt.close()
    
    # 6. Biểu đồ tương quan giữa các metrics (nếu có nhiều hơn 1 metric)
    numeric_metrics = [col for col in metrics_columns if df[col].dtype in [int, float]]
    if len(numeric_metrics) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_metrics].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Ma trận tương quan giữa các Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'metrics_correlation.png'))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Theo dõi và phân tích metrics từ MLflow")
    parser.add_argument("--days", type=int, default=7, help="Số ngày dữ liệu cần phân tích")
    parser.add_argument("--output-dir", type=str, default="metrics_reports", help="Thư mục để lưu báo cáo")
    
    args = parser.parse_args()
    track_metrics(args.days, args.output_dir)