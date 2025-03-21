from ultralytics import YOLO
import numpy as np
import os

# Điều kiện hóa việc import pytorch và mlflow
TRAINING_MODE = False  # Đặt thành False khi triển khai, True khi huấn luyện

if TRAINING_MODE:
    import torch
    import mlflow
    import mlflow.pytorch

# Kiểm tra xem onnxruntime có được cài đặt không
try:
    import onnxruntime as ort
    import cv2
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

def load_model(model_path='yolo11s.pt'):
    """Tải mô hình YOLOv11s.
    
    Args:
        model_path (str): Đường dẫn đến trọng số mô hình
        
    Returns:
        YOLO or ONNXModel: Mô hình đã tải
    """
    try:
        # Kiểm tra xem mô hình có định dạng ONNX và onnxruntime có sẵn không
        if model_path.endswith('.onnx') and ONNX_AVAILABLE:
            return ONNXModel(model_path)
        else:
            model = YOLO(model_path)
            return model
    except Exception as e:
        raise Exception(f"Lỗi khi tải mô hình: {str(e)}")

def train_model(data_yaml, epochs=100, imgsz=640, batch_size=16, resume=False, resume_path=None, run_id=None):
    """Huấn luyện mô hình YOLOv11s với các tham số được chỉ định.

    Args:
        data_yaml (str): Đường dẫn đến file cấu hình dữ liệu YAML
        epochs (int): Số lượng epoch huấn luyện
        imgsz (int): Kích thước ảnh đầu vào
        batch_size (int): Kích thước batch cho huấn luyện
        resume (bool): Có tiếp tục huấn luyện từ checkpoint không
        resume_path (str): Đường dẫn đến checkpoint để tiếp tục
        run_id (str): ID của MLflow run để tiếp tục ghi log

    Returns:
        tuple: Kết quả huấn luyện và run ID
    """
    try:
        # Khởi tạo mô hình
        if resume and resume_path:
            model = YOLO(resume_path)
        elif resume:
            model = YOLO('models/train/weights/last.pt')
        else:
            model = YOLO('yolo11s.pt')

        # Thiết lập MLflow tracking
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Nếu run_id được cung cấp, lấy experiment ID cho run đó
        if run_id:
            try:
                run_info = mlflow.get_run(run_id)
                experiment_id = run_info.info.experiment_id
                experiment = mlflow.get_experiment(experiment_id)
                mlflow.set_experiment(experiment.name)
            except Exception as e:
                print(f"Cảnh báo: Không thể thiết lập experiment từ run_id: {str(e)}")
                print("Tạo một run mới thay thế.")
                run_id = None
        else:
            mlflow.set_experiment("brain_tumor_detection")

        # Bật tự động ghi log
        mlflow.pytorch.autolog()

        # Bắt đầu hoặc tiếp tục MLflow run
        if run_id:
            with mlflow.start_run(run_id=run_id) as run:
                # Huấn luyện mô hình
                results = model.train(
                    data=data_yaml,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    device=0 if torch.cuda.is_available() else 'cpu',
                    verbose=False,
                    project='models',
                    name=None,
                    exist_ok=True,
                    resume=resume
                )
                return results, run.info.run_id
        else:
            with mlflow.start_run() as run:
                # Huấn luyện mô hình
                results = model.train(
                    data=data_yaml,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    device=0 if torch.cuda.is_available() else 'cpu',
                    verbose=False,
                    project='models',
                    name=None,
                    exist_ok=True,
                    resume=resume
                )
                return results, run.info.run_id

    except Exception as e:
        raise Exception(f"Lỗi trong quá trình huấn luyện: {str(e)}")

def predict(model, image, conf_thres=0.25):
    """Thực hiện dự đoán trên một ảnh.

    Args:
        model: Mô hình YOLO hoặc ONNXModel đã tải
        image: Ảnh đầu vào (mảng numpy)
        conf_thres (float): Ngưỡng tin cậy

    Returns:
        Results: Kết quả dự đoán của YOLO
    """
    try:
        results = model(image, conf=conf_thres)
        return results
    except Exception as e:
        import traceback
        print(f"Lỗi chi tiết trong hàm dự đoán:")
        traceback.print_exc()
        raise Exception(f"Lỗi trong quá trình dự đoán: {str(e)}")

# Thêm lớp ONNXModel chỉ khi onnxruntime có sẵn
if ONNX_AVAILABLE:
    class ONNXModel:
        """Wrapper cho mô hình ONNX Runtime để duy trình tính tương thích với API của YOLO"""
        
        # Định nghĩa các lớp Box, Boxes và Results ở cấp độ lớp
        class Box:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = xyxy
                self.conf = conf
                self.cls = cls
        
        class Boxes:
            def __init__(self):
                self.xyxy = []
                self.conf = []
                self.cls = []
                self._len = 0
                
            def append(self, box):
                try:
                    # Xử lý xyxy
                    if isinstance(box.xyxy, np.ndarray):
                        self.xyxy.append(box.xyxy)
                    else:
                        self.xyxy.append(np.array([box.xyxy]))
                        
                    # Xử lý conf
                    if isinstance(box.conf, np.ndarray):
                        self.conf.append(box.conf)
                    else:
                        self.conf.append(np.array([box.conf]))
                        
                    # Xử lý cls
                    if isinstance(box.cls, np.ndarray):
                        self.cls.append(box.cls)
                    else:
                        self.cls.append(np.array([box.cls]))
                        
                    self._len += 1
                except Exception as e:
                    print(f"Lỗi khi thêm box: {str(e)}")
            
            def __len__(self):
                return self._len
            
            def __iter__(self):
                # Tạo một iterator giả để hỗ trợ lặp qua boxes
                class BoxIterator:
                    def __init__(self, boxes):
                        self.boxes = boxes
                        self.index = 0
                        
                    def __iter__(self):
                        return self
                        
                    def __next__(self):
                        if self.index < len(self.boxes):
                            try:
                                # Đảm bảo dữ liệu có định dạng đúng
                                xyxy = self.boxes.xyxy[self.index]
                                conf = self.boxes.conf[self.index]
                                cls = self.boxes.cls[self.index]
                                
                                # Kiểm tra và chuyển đổi dữ liệu nếu cần
                                if isinstance(xyxy, list):
                                    xyxy = np.array(xyxy)
                                if isinstance(conf, list):
                                    conf = np.array(conf)
                                if isinstance(cls, list):
                                    cls = np.array(cls)
                                    
                                box = ONNXModel.Box(xyxy=xyxy, conf=conf, cls=cls)
                                self.index += 1
                                return box
                            except Exception as e:
                                print(f"Lỗi trong __next__ tại index {self.index}: {str(e)}")
                                self.index += 1
                                # Nếu có lỗi, thử box tiếp theo
                                return self.__next__()
                        raise StopIteration
                
                return BoxIterator(self)
        
        class Results:
            def __init__(self, boxes, names):
                self.boxes = boxes
                self.names = names
        
        def __init__(self, model_path):
            """Khởi tạo mô hình ONNX
            
            Args:
                model_path (str): Đường dẫn đến mô hình ONNX
            """
            # Kiểm tra xem mô hình ONNX có tồn tại không
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy mô hình ONNX: {model_path}")
            
            # Tạo session ONNX Runtime
            self.session = ort.InferenceSession(
                model_path, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Lấy metadata của mô hình
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Lấy shape đầu vào
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Đảm bảo img_size là tuple với các số nguyên
            try:
                height = int(self.input_shape[-2]) if isinstance(self.input_shape[-2], (int, float, str)) else 640
                width = int(self.input_shape[-1]) if isinstance(self.input_shape[-1], (int, float, str)) else 640
                self.img_size = (height, width)  # Height, width
            except (ValueError, TypeError, IndexError):
                # Fallback to default size if there's any error
                self.img_size = (640, 640)
                print("Cảnh báo: Không thể xác định kích thước đầu vào từ mô hình ONNX, sử dụng mặc định (640, 640)")
            
            # Tên các lớp cho phát hiện khối u não
            self.names = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
        
        def __call__(self, img, conf=0.25):
            """Chạy dự đoán trên ảnh
            
            Args:
                img: Ảnh đầu vào (mảng numpy)
                conf (float): Ngưỡng tin cậy
                
            Returns:
                Results: Kết quả tương thích với YOLO
            """
            # Tiền xử lý ảnh
            input_data = self._preprocess(img)
            
            # Chạy dự đoán
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            # Hậu xử lý kết quả
            results = self._postprocess(outputs, img, conf)
            
            return results
        
        def _preprocess(self, img):
            """Tiền xử lý ảnh cho mô hình ONNX
            
            Args:
                img: Ảnh đầu vào
                
            Returns:
                numpy.ndarray: Ảnh đã xử lý
            """
            # Chuyển đổi sang RGB nếu cần
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Resize - Đảm bảo kích thước là tuple với các số nguyên
            # Kiểm tra và đảm bảo self.img_size chứa các số nguyên
            try:
                width = int(self.img_size[1])
                height = int(self.img_size[0])
            except (ValueError, TypeError):
                # Nếu không thể chuyển đổi, sử dụng kích thước mặc định
                width, height = 640, 640
                
            img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Chuẩn hóa
            img_norm = img_resized.astype(np.float32) / 255.0
            
            # HWC sang NCHW
            img_transposed = img_norm.transpose(2, 0, 1)
            
            # Thêm chiều batch
            img_batch = np.expand_dims(img_transposed, axis=0)
            
            return img_batch
        
        def _postprocess(self, outputs, original_img, conf_threshold):
            """Hậu xử lý đầu ra của mô hình ONNX
            
            Args:
                outputs: Đầu ra của mô hình ONNX
                original_img: Ảnh đầu vào gốc
                conf_threshold: Ngưỡng tin cậy
                
            Returns:
                list: Danh sách các đối tượng Results
            """
            # Xử lý các phát hiện
            detections = outputs[0]
            
            # Debug: In thông tin về shape
            print(f"DEBUG - Output shape: {detections.shape}")
            print(f"DEBUG - Output type: {type(detections)}")
            
            # Khởi tạo boxes
            boxes = self.Boxes()
            
            try:
                # Xử lý đầu ra dạng (1, 8, 8400) - định dạng phổ biến của YOLOv8 ONNX
                if len(detections.shape) == 3:
                    print(f"DEBUG - Đang xử lý đầu ra 3D với shape {detections.shape}")
                    
                    # Chuyển đổi từ [1, 8, 8400] sang [8400, 8] để dễ xử lý
                    # Đảm bảo chuyển đổi đúng chiều
                    if detections.shape[1] == 8 and detections.shape[2] == 8400:
                        # Nếu shape là [1, 8, 8400]
                        detections = detections[0]  # Giờ shape là [8, 8400]
                        confidence = detections[4, :]  # Lấy confidence từ hàng thứ 5
                    elif detections.shape[1] == 8400 and detections.shape[2] == 8:
                        # Nếu shape là [1, 8400, 8]
                        detections = detections[0].transpose()  # Giờ shape là [8, 8400]
                        confidence = detections[4, :]  # Lấy confidence từ hàng thứ 5
                    else:
                        # Nếu shape khác, thử chuyển đổi
                        print(f"DEBUG - Shape không thông thường, đang cố gắng điều chỉnh")
                        detections = detections.reshape(-1, detections.shape[-1])
                        if detections.shape[1] >= 5:
                            confidence = detections[:, 4]
                        else:
                            print("DEBUG - Không thể trích xuất confidence, shape không tương thích")
                            return [Results(boxes, self.names)]
                    
                    print(f"DEBUG - Confidence shape: {confidence.shape}")
                    print(f"DEBUG - Confidence min: {np.min(confidence)}, max: {np.max(confidence)}")
                    
                    # Lọc theo ngưỡng confidence
                    mask = confidence >= conf_threshold
                    print(f"DEBUG - Mask sum: {np.sum(mask)}")
                    
                    if np.any(mask):
                        # Tiếp tục xử lý như trước
                        # ...
                        # Lấy các dự đoán vượt qua ngưỡng
                        filtered_detections = detections[:, mask]
                        
                        # Lấy tọa độ, confidence và class
                        boxes_data = filtered_detections[:4, :].transpose(1, 0)  # [x, y, w, h]
                        scores = filtered_detections[4, :]
                        
                        # Xử lý class scores
                        if filtered_detections.shape[0] > 5:
                            class_scores = filtered_detections[5:, :]
                            class_ids = np.argmax(class_scores, axis=0)
                        else:
                            # Nếu không có class scores, giả định class 0
                            class_ids = np.zeros(scores.shape, dtype=np.int32)
                        
                        # Chuyển đổi từ [x, y, w, h] sang [x1, y1, x2, y2]
                        x1y1 = boxes_data[:, :2] - boxes_data[:, 2:] / 2
                        x2y2 = boxes_data[:, :2] + boxes_data[:, 2:] / 2
                        boxes_xyxy = np.hstack((x1y1, x2y2))
                        
                        # Scale về kích thước ảnh gốc
                        orig_h, orig_w = original_img.shape[:2]
                        scale_x = orig_w / self.img_size[1]
                        scale_y = orig_h / self.img_size[0]
                        
                        # Áp dụng scale
                        boxes_xyxy[:, 0] *= scale_x
                        boxes_xyxy[:, 1] *= scale_y
                        boxes_xyxy[:, 2] *= scale_x
                        boxes_xyxy[:, 3] *= scale_y
                        
                        # Tạo các box và thêm vào danh sách
                        for i in range(len(scores)):
                            box = self.Box(
                                xyxy=np.array([boxes_xyxy[i]]),
                                conf=np.array([scores[i]]),
                                cls=np.array([class_ids[i]])
                            )
                            boxes.append(box)
                        
                        # Thêm bước Non-Maximum Suppression để loại bỏ các box trùng lặp
                        boxes = self._apply_nms(boxes, iou_threshold=0.45)
                        
                    else:
                        print("DEBUG - Định dạng đầu ra không được hỗ trợ")
            except Exception as e:
                print(f"DEBUG - Lỗi trong hậu xử lý: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Tạo và trả về̉ kết quả
            results = [self.Results(boxes, self.names)]
            return results
        
        def _apply_nms(self, boxes, iou_threshold=0.45):
            """Áp dụng Non-Maximum Suppression để loại bỏ các box trùng lặp
            
            Args:
                boxes: Đối tượng Boxes chứa các box phát hiện
                iou_threshold: Ngưỡng IoU để xem xét các box là trùng lặp
                
            Returns:
                Boxes: Các box đã lọc sau NMS
            """
            if len(boxes) <= 1:
                return boxes
                
            # Tạo Boxes mới để lưu kết quả sau NMS
            filtered_boxes = self.Boxes()
            
            # Chuyển đổi danh sách boxes thành mảng numpy để dễ xử lý
            all_xyxy = []
            all_conf = []
            all_cls = []
            
            # Thu thập tất cả boxes
            for box in boxes:
                all_xyxy.append(box.xyxy[0])  # Lấy tọa độ box
                all_conf.append(box.conf[0])  # Lấy confidence
                all_cls.append(box.cls[0])    # Lấy class id
            
            # Chuyển thành mảng numpy
            all_xyxy = np.array(all_xyxy)
            all_conf = np.array(all_conf)
            all_cls = np.array(all_cls)
            
            # Sắp xếp theo độ tin cậy giảm dần
            indices = np.argsort(-all_conf)
            
            # Áp dụng NMS
            keep_indices = []
            while len(indices) > 0:
                # Lấy box có độ tin cậy cao nhất
                current_idx = indices[0]
                keep_indices.append(current_idx)
                
                # Nếu chỉ còn 1 box, thoát vòng lặp
                if len(indices) == 1:
                    break
                
                # Loại bỏ box hiện tại khỏi danh sách
                indices = indices[1:]
                
                # Tính IoU giữa box hiện tại và các box còn lại
                current_box = all_xyxy[current_idx]
                remaining_boxes = all_xyxy[indices]
                
                # Tính diện tích các box
                current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
                
                # Tính tọa độ giao nhau
                xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
                yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
                xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
                yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])
                
                # Tính diện tích phần giao nhau
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                intersection = w * h
                
                # Tính IoU
                union = current_area + remaining_areas - intersection
                iou = intersection / (union + 1e-6)  # Thêm epsilon để tránh chia cho 0
                
                # Lọc ra các box có IoU nhỏ hơn ngưỡng
                # Chỉ giữ lại các box có cùng class với box hiện tại
                same_class = all_cls[indices] == all_cls[current_idx]
                low_iou = iou < iou_threshold
                mask = np.logical_or(low_iou, ~same_class)
                indices = indices[mask]
            
            # Tạo boxes mới từ các indices đã giữ lại
            for idx in keep_indices:
                box = self.Box(
                    xyxy=np.array([all_xyxy[idx]]),
                    conf=np.array([all_conf[idx]]),
                    cls=np.array([all_cls[idx]])
                )
                filtered_boxes.append(box)
            
            return filtered_boxes
