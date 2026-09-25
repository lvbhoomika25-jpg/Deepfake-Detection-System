

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================================
# DEEPFAKE DETECTOR CLASS
# ============================================================================

class AdvancedDeepfakeDetector:
    """Advanced deepfake detection with multiple algorithms"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize ML models
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_trained = False

        # Detection settings
        self.settings = {
            'confidence_threshold': 0.7,
            'face_detection_confidence': 0.5,
            'use_face_recognition': True,
            'use_mediapipe': True,
            'use_ml_classifier': True,
            'analysis_depth': 'high'  # low, medium, high
        }

    def extract_advanced_features(self, image):
        """Extract comprehensive features for deepfake detection"""
        features = {}

        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Face Landmark Analysis (simplified without MediaPipe)
        features.update(self._analyze_face_landmarks(rgb_image))

        # 2. Frequency Domain Analysis
        features.update(self._frequency_analysis(rgb_image))

        # 3. Texture Analysis
        features.update(self._texture_analysis(rgb_image))

        # 4. Color Analysis
        features.update(self._color_analysis(rgb_image))

        # 5. Edge Analysis
        features.update(self._edge_analysis(rgb_image))

        # 6. Compression Artifact Detection
        features.update(self._compression_analysis(rgb_image))

        return features

    def _analyze_face_landmarks(self, image):
        """Analyze facial landmarks for inconsistencies (simplified version)"""
        features = {}

        # Detect faces using OpenCV
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Use largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face

            # Basic face analysis
            features['face_aspect_ratio'] = w / h

            # Simple symmetry check using face center
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            features['face_center_x'] = face_center_x / image.shape[1]
            features['face_center_y'] = face_center_y / image.shape[0]

            # Face size relative to image
            features['face_size_ratio'] = (w * h) / (image.shape[0] * image.shape[1])

        return features

    def _frequency_analysis(self, image):
        """Analyze frequency domain for artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # FFT analysis
        f_transform = np.fft.fft2(gray)
        magnitude_spectrum = np.abs(f_transform)

        # High frequency energy
        h, w = magnitude_spectrum.shape
        high_freq_energy = np.mean(magnitude_spectrum[h // 4:, w // 4:])
        low_freq_energy = np.mean(magnitude_spectrum[:h // 4, :w // 4])

        return {
            'high_freq_ratio': high_freq_energy / (low_freq_energy + 1e-6),
            'spectral_centroid': np.sum(magnitude_spectrum * np.arange(h)[:, None]) / np.sum(magnitude_spectrum)
        }

    def _texture_analysis(self, image):
        """Advanced texture analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Local Binary Pattern approximation
        lbp_features = []
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                binary_string = ""
                for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
                    binary_string += "1" if gray[i + di, j + dj] >= center else "0"
                lbp_features.append(int(binary_string, 2))

        return {
            'texture_uniformity': np.std(lbp_features),
            'texture_entropy': -np.sum(np.bincount(lbp_features) * np.log2(np.bincount(lbp_features) + 1e-6))
        }

    def _color_analysis(self, image):
        """Advanced color analysis"""
        # HSV analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Skin tone analysis
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = np.sum(skin_mask > 0)

        # Color consistency
        color_variance = np.var(image.reshape(-1, 3), axis=0)

        return {
            'skin_tone_consistency': skin_pixels / image.size,
            'color_variance_r': color_variance[0],
            'color_variance_g': color_variance[1],
            'color_variance_b': color_variance[2]
        }

    def _edge_analysis(self, image):
        """Advanced edge analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Multiple edge detection methods
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1)

        # Edge density
        edge_density = np.sum(edges_canny > 0) / edges_canny.size

        # Edge direction analysis
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        edge_directions = np.arctan2(sobel_y, sobel_x)

        return {
            'edge_density': edge_density,
            'edge_direction_consistency': np.std(edge_directions),
            'edge_sharpness': np.mean(np.abs(edges_sobel))
        }

    def _compression_analysis(self, image):
        """Detect compression artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # DCT analysis for JPEG artifacts
        block_size = 8
        h, w = gray.shape

        dct_blocks = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i + block_size, j:j + block_size].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_blocks.append(dct_block)

        if dct_blocks:
            dct_blocks = np.array(dct_blocks)
            # High frequency DCT coefficients indicate compression
            high_freq_energy = np.mean(np.abs(dct_blocks[:, 4:, 4:]))

            return {'compression_artifacts': high_freq_energy}

        return {'compression_artifacts': 0}

    def detect_deepfake(self, image_path):
        """Main detection function - handles both images and videos"""
        try:
            # Handle video inputs by delegating to video pipeline
            if image_path.lower().endswith((
                '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'
            )):
                return self._detect_deepfake_video(image_path)

            image = cv2.imread(image_path)
            if image is None:
                return "ERROR", "Could not load image", {}

            # Detect faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                return "NO_FACE", "No face detected", {}

            # Analyze largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face_roi = image[y:y + h, x:x + w]

            # Extract features
            features = self.extract_advanced_features(face_roi)

            # Calculate deepfake probability
            fake_probability = self._calculate_fake_probability(features)

            # Determine result
            if fake_probability > self.settings['confidence_threshold']:
                result = "FAKE"
            elif fake_probability < (1 - self.settings['confidence_threshold']):
                result = "REAL"
            else:
                result = "UNCERTAIN"

            confidence = max(fake_probability, 1 - fake_probability)

            return result, confidence, features

        except Exception as e:
            return "ERROR", str(e), {}

    def _detect_deepfake_video(self, video_path):
        """Detect deepfake in a video by sampling frames and aggregating results"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "ERROR", "Could not open video", {}
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            duration_s = frame_count / fps if fps else 0
            # Decide sampling: up to 32 evenly spaced frames
            max_samples = 32
            num_samples = min(max_samples, frame_count if frame_count > 0 else max_samples)
            indices = np.linspace(0, max(0, frame_count - 1), int(num_samples)).astype(int) if frame_count > 0 else np.arange(max_samples)
            fake_probs = []
            valid_samples = 0
            for idx in indices:
                if frame_count > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) == 0:
                    continue
                # Use largest face region
                x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
                face_roi = frame[y:y + h, x:x + w]
                features = self.extract_advanced_features(face_roi)
                prob = self._calculate_fake_probability(features)
                fake_probs.append(float(prob))
                valid_samples += 1
            if valid_samples == 0:
                # Fall back: try center crop if no faces detected
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    return "NO_FACE", "No readable frames", {
                        'frames_total': frame_count,
                        'fps': fps,
                        'duration_s': duration_s
                    }
                h, w, _ = frame.shape
                ch, cw = h // 2, w // 2
                size = min(ch, cw)
                center_crop = frame[ch - size:ch + size, cw - size:cw + size]
                features = self.extract_advanced_features(center_crop)
                prob = self._calculate_fake_probability(features)
                fake_probs.append(float(prob))
                valid_samples = 1
            avg_prob = float(np.mean(fake_probs)) if fake_probs else 0.5
            confidence = max(avg_prob, 1 - avg_prob)
            if avg_prob > self.settings['confidence_threshold']:
                result = "FAKE"
            elif avg_prob < (1 - self.settings['confidence_threshold']):
                result = "REAL"
            else:
                result = "UNCERTAIN"
            features_summary = {
                'frames_total': frame_count,
                'frames_sampled': int(num_samples),
                'frames_valid': valid_samples,
                'fps': fps,
                'duration_s': duration_s,
                'avg_fake_probability': avg_prob
            }
            return result, confidence, features_summary
        finally:
            cap.release()

    def _calculate_fake_probability(self, features):
        """Calculate probability that content is fake"""
        if not self.model_trained:
            # Use rule-based approach if model not trained
            score = 0
            total_checks = 0

            # Eye symmetry check
            if 'eye_symmetry' in features:
                if features['eye_symmetry'] > 0.1:
                    score += 1
                total_checks += 1

            # Face aspect ratio check
            if 'face_aspect_ratio' in features:
                if not (0.6 <= features['face_aspect_ratio'] <= 1.0):
                    score += 1
                total_checks += 1

            # High frequency analysis
            if 'high_freq_ratio' in features:
                if features['high_freq_ratio'] > 2.0:
                    score += 1
                total_checks += 1

            # Texture analysis
            if 'texture_uniformity' in features:
                if features['texture_uniformity'] < 50:
                    score += 1
                total_checks += 1

            # Color analysis
            if 'skin_tone_consistency' in features:
                if features['skin_tone_consistency'] < 0.3:
                    score += 1
                total_checks += 1

            if total_checks == 0:
                return 0.5  # Uncertain if no features

            fake_probability = score / total_checks
            return fake_probability
        else:
            # Use trained ML model
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            feature_vector = self.scaler.transform(feature_vector)
            probability = self.classifier.predict_proba(feature_vector)[0][1]
            return probability


# ============================================================================
# GUI CLASS
# ============================================================================

class ModernDeepfakeGUI:
    """Modern GUI for deepfake detection"""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Advanced Deepfake Detection System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize detector
        self.detector = AdvancedDeepfakeDetector()
        
        # Variables
        self.current_file = None
        self.current_image = None
        self.current_video = None
        self.is_video = False
        self.video_cap = None
        self.analysis_results = {}
        
        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.create_sidebar()
        self.create_main_area()
        self.create_status_bar()
    
    def create_sidebar(self):
        """Create sidebar with controls"""
        self.sidebar = ctk.CTkFrame(self.root, width=300)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        # Title
        title = ctk.CTkLabel(self.sidebar, text="🔍 Deepfake Detector", 
                           font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=(20, 30))
        
        # File selection
        file_frame = ctk.CTkFrame(self.sidebar)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(file_frame, text="📁 Select Media", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        ctk.CTkLabel(file_frame, text="(Images & Videos)", 
                    font=ctk.CTkFont(size=12),
                    text_color="gray").pack(pady=(0, 10))
        
        self.browse_btn = ctk.CTkButton(file_frame, text="Browse Files", 
                                      command=self.browse_file,
                                      height=40)
        self.browse_btn.pack(pady=10)
        
        self.drag_label = ctk.CTkLabel(file_frame, text="or drag & drop here",
                                     text_color="gray")
        self.drag_label.pack(pady=(0, 10))
        
        # Detection settings
        settings_frame = ctk.CTkFrame(self.sidebar)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(settings_frame, text="⚙️ Settings", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Confidence threshold
        ctk.CTkLabel(settings_frame, text="Confidence Threshold:").pack()
        self.confidence_slider = ctk.CTkSlider(settings_frame, from_=0.5, to=0.95, 
                                             command=self.update_confidence)
        self.confidence_slider.set(0.7)
        self.confidence_slider.pack(pady=5)
        
        self.confidence_label = ctk.CTkLabel(settings_frame, text="70%")
        self.confidence_label.pack()
        
        # Analysis depth
        ctk.CTkLabel(settings_frame, text="Analysis Depth:").pack(pady=(10, 0))
        self.depth_option = ctk.CTkOptionMenu(settings_frame, 
                                            values=["Low", "Medium", "High"],
                                            command=self.update_depth)
        self.depth_option.set("High")
        self.depth_option.pack(pady=5)
        
        # Detection options
        self.use_advanced_analysis = ctk.CTkCheckBox(settings_frame, text="Advanced Analysis")
        self.use_advanced_analysis.pack(pady=5)
        
        # Action buttons
        action_frame = ctk.CTkFrame(self.sidebar)
        action_frame.pack(fill="x", padx=20, pady=20)
        
        self.detect_btn = ctk.CTkButton(action_frame, text="🔍 Detect Deepfake", 
                                      command=self.start_detection,
                                      height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.detect_btn.pack(pady=10)
        
        self.batch_btn = ctk.CTkButton(action_frame, text="📁 Batch Process", 
                                     command=self.batch_process,
                                     height=40)
        self.batch_btn.pack(pady=5)
        
        self.clear_btn = ctk.CTkButton(action_frame, text="🗑️ Clear", 
                                     command=self.clear_results,
                                     height=40, fg_color="gray")
        self.clear_btn.pack(pady=5)
    
    def create_main_area(self):
        """Create main display area"""
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Image preview tab
        self.preview_tab = self.notebook.add("📷 Preview")
        self.notebook.tab("📷 Preview").grid_columnconfigure(0, weight=1)
        self.notebook.tab("📷 Preview").grid_rowconfigure(0, weight=1)
        
        self.preview_frame = ctk.CTkFrame(self.notebook.tab("📷 Preview"))
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(self.preview_frame, text="No image selected")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # Analysis tab
        self.analysis_tab = self.notebook.add("📊 Analysis")
        self.notebook.tab("📊 Analysis").grid_columnconfigure(0, weight=1)
        self.notebook.tab("📊 Analysis").grid_rowconfigure(0, weight=1)
        
        self.analysis_frame = ctk.CTkScrollableFrame(self.notebook.tab("📊 Analysis"))
        self.analysis_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Results tab
        self.results_tab = self.notebook.add("📋 Results")
        self.notebook.tab("📋 Results").grid_columnconfigure(0, weight=1)
        self.notebook.tab("📋 Results").grid_rowconfigure(0, weight=1)
        
        self.results_frame = ctk.CTkScrollableFrame(self.notebook.tab("📋 Results"))
        self.results_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.progress_bar.set(0)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ctk.CTkFrame(self.root, height=30)
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready to detect deepfakes (images & videos)")
        self.status_label.grid(row=0, column=0, sticky="w", padx=10)
        
        self.time_label = ctk.CTkLabel(self.status_frame, text="")
        self.time_label.grid(row=0, column=1, sticky="e", padx=10)
        
        # Update time
        self.update_time()
    
    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.configure(text=current_time)
        self.root.after(1000, self.update_time)
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        self.detector.settings['confidence_threshold'] = value
        self.confidence_label.configure(text=f"{int(value*100)}%")
    
    def update_depth(self, value):
        """Update analysis depth"""
        self.detector.settings['analysis_depth'] = value.lower()
    
    def browse_file(self):
        """Browse for files"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=file_types
        )
        
        if filename:
            self.load_file(filename)
    
    def load_file(self, filepath):
        """Load and display file (image or video)"""
        self.current_file = filepath
        self.status_label.configure(text=f"Loading: {os.path.basename(filepath)}")
        
        # Check if video or image
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        try:
            if filepath.lower().endswith(image_extensions):
                # Handle image file
                self.is_video = False
                if self.video_cap:
                    self.video_cap.release()
                    self.video_cap = None
                
                image = Image.open(filepath)
                image.thumbnail((600, 400), Image.Resampling.LANCZOS)
                self.current_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
                self.image_label.configure(image=self.current_image, text="")
                self.status_label.configure(text=f"Image loaded: {os.path.basename(filepath)}")
                
            elif filepath.lower().endswith(video_extensions):
                # Handle video file
                self.is_video = True
                self.load_video_preview(filepath)
                
            else:
                messagebox.showwarning("Warning", "Unsupported file format")
                self.status_label.configure(text="Unsupported file format")
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")
            self.status_label.configure(text="Error loading file")
    
    def load_video_preview(self, filepath):
        """Load and display video preview"""
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Read first frame for preview
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize for preview
                frame_pil.thumbnail((600, 400), Image.Resampling.LANCZOS)
                self.current_image = ctk.CTkImage(light_image=frame_pil, dark_image=frame_pil, size=frame_pil.size)
                
                # Update preview with video info
                video_info = f"📹 Video Preview\n"
                video_info += f"Duration: {duration:.1f}s\n"
                video_info += f"Resolution: {width}x{height}\n"
                video_info += f"FPS: {fps:.1f}\n"
                video_info += f"Frames: {frame_count}"
                
                self.image_label.configure(image=self.current_image, text=video_info, compound="bottom")
                self.status_label.configure(text=f"Video loaded: {os.path.basename(filepath)}")
                
                # Store video cap for later use
                self.video_cap = cap
                self.current_video = {
                    'path': filepath,
                    'fps': fps,
                    'frame_count': frame_count,
                    'width': width,
                    'height': height,
                    'duration': duration
                }
            else:
                cap.release()
                raise Exception("Could not read video frames")
                
        except Exception as e:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            self.image_label.configure(image=None, text=f"Video file selected\nError loading preview: {str(e)}")
            self.status_label.configure(text="Video file loaded (preview unavailable)")
            self.current_video = {'path': filepath}
    
    def start_detection(self):
        """Start detection process"""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        # Update settings from GUI
        self.detector.settings['use_advanced_analysis'] = self.use_advanced_analysis.get()
        
        # Start detection in thread
        self.progress_bar.set(0)
        self.detect_btn.configure(state="disabled")
        
        thread = threading.Thread(target=self.run_detection)
        thread.daemon = True
        thread.start()
    
    def run_detection(self):
        """Run detection in separate thread (supports both images and videos)"""
        try:
            file_type = "video" if self.is_video else "image"
            self.root.after(0, lambda: self.status_label.configure(text=f"Analyzing {file_type}..."))
            self.root.after(0, lambda: self.progress_bar.set(0.2))
            
            # Run detection (detector automatically handles images and videos)
            result, confidence, features = self.detector.detect_deepfake(self.current_file)
            
            self.root.after(0, lambda: self.progress_bar.set(0.8))
            
            # Store results
            self.analysis_results = {
                'result': result,
                'confidence': confidence,
                'features': features,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': os.path.basename(self.current_file),
                'file_type': file_type
            }
            
            # Add video-specific info if available
            if self.is_video and self.current_video:
                self.analysis_results['video_info'] = self.current_video
            
            # Update GUI
            self.root.after(0, self.update_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            self.root.after(0, lambda: self.detect_btn.configure(state="normal"))
            self.root.after(0, lambda: self.status_label.configure(text="Ready"))
    
    def update_results(self):
        """Update results display (supports both images and videos)"""
        if not self.analysis_results:
            return
        
        # Switch to results tab
        self.notebook.set("📋 Results")
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Display results
        result = self.analysis_results['result']
        confidence = self.analysis_results['confidence']
        file_type = self.analysis_results.get('file_type', 'image')
        
        # Main result
        result_frame = ctk.CTkFrame(self.results_frame)
        result_frame.pack(fill="x", padx=10, pady=10)
        
        if result == "FAKE":
            color = "red"
            icon = "⚠️"
            message = "DEEPFAKE DETECTED"
        elif result == "REAL":
            color = "green"
            icon = "✅"
            message = "AUTHENTIC CONTENT"
        elif result == "NO_FACE":
            color = "orange"
            icon = "⚠️"
            message = "NO FACE DETECTED"
        else:
            color = "orange"
            icon = "❓"
            message = "UNCERTAIN"
        
        # File type indicator
        type_icon = "📹" if file_type == "video" else "📷"
        ctk.CTkLabel(result_frame, text=f"{type_icon} {file_type.upper()} ANALYSIS", 
                    font=ctk.CTkFont(size=14, weight="bold"),
                    text_color="gray").pack(pady=(10, 5))
        
        ctk.CTkLabel(result_frame, text=f"{icon} {message}", 
                    font=ctk.CTkFont(size=24, weight="bold"),
                    text_color=color).pack(pady=20)
        
        ctk.CTkLabel(result_frame, text=f"Confidence: {confidence:.1%}", 
                    font=ctk.CTkFont(size=18)).pack(pady=10)
        
        # Video-specific information
        if file_type == "video" and 'video_info' in self.analysis_results:
            self.display_video_info()
        
        # Detailed analysis
        if self.analysis_results['features']:
            self.display_detailed_analysis()
        
        # Show message box
        file_name = self.analysis_results.get('filename', 'file')
        if result == "FAKE":
            messagebox.showwarning("Deepfake Detected", 
                                 f"The {file_type} appears to be manipulated!\n\n"
                                 f"File: {file_name}\n"
                                 f"Confidence: {confidence:.1%}")
        elif result == "REAL":
            messagebox.showinfo("Authentic Content", 
                              f"The {file_type} appears to be authentic!\n\n"
                              f"File: {file_name}\n"
                              f"Confidence: {confidence:.1%}")
        elif result == "NO_FACE":
            messagebox.showwarning("No Face Detected", 
                                 f"No face was detected in the {file_type}.\n"
                                 f"Please select a {file_type} containing a visible face.")
    
    def display_video_info(self):
        """Display video-specific information"""
        if 'video_info' not in self.analysis_results:
            return
        
        video_info = self.analysis_results['video_info']
        
        info_frame = ctk.CTkFrame(self.results_frame)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(info_frame, text="📹 Video Information", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        info_text = f"Duration: {video_info.get('duration', 0):.1f} seconds\n"
        info_text += f"Resolution: {video_info.get('width', 0)}x{video_info.get('height', 0)}\n"
        info_text += f"Frame Rate: {video_info.get('fps', 0):.1f} FPS\n"
        info_text += f"Total Frames: {video_info.get('frame_count', 0)}"
        
        if 'frames_sampled' in self.analysis_results.get('features', {}):
            info_text += f"\nFrames Analyzed: {self.analysis_results['features'].get('frames_valid', 0)}"
        
        ctk.CTkLabel(info_frame, text=info_text, 
                    font=ctk.CTkFont(size=12), justify="left").pack(pady=5)
    
    def display_detailed_analysis(self):
        """Display detailed analysis results"""
        features = self.analysis_results['features']
        
        # Features analysis
        features_frame = ctk.CTkFrame(self.results_frame)
        features_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(features_frame, text="🔬 Detailed Analysis", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Create feature grid
        for i, (key, value) in enumerate(features.items()):
            if isinstance(value, (int, float)):
                feature_frame = ctk.CTkFrame(features_frame)
                feature_frame.pack(fill="x", padx=10, pady=5)
                
                ctk.CTkLabel(feature_frame, text=f"{key.replace('_', ' ').title()}: {value:.4f}").pack(side="left", padx=10, pady=5)
        
        # Create visualization
        self.create_analysis_chart()
    
    def create_analysis_chart(self):
        """Create analysis visualization"""
        features = self.analysis_results['features']
        
        # Filter numeric features
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        
        if len(numeric_features) < 2:
            return
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart of features
        feature_names = list(numeric_features.keys())
        feature_values = list(numeric_features.values())
        
        ax1.bar(range(len(feature_names)), feature_values, color='skyblue')
        ax1.set_title('Feature Analysis')
        ax1.set_xticks(range(len(feature_names)))
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in feature_names], rotation=45)
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        values = feature_values + feature_values[:1]  # Close the circle
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values, 'o-', linewidth=2, color='red')
        ax2.fill(angles, values, alpha=0.25, color='red')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([name.replace('_', ' ').title() for name in feature_names])
        ax2.set_title('Feature Radar')
        
        plt.tight_layout()
        
        # Embed in GUI
        chart_frame = ctk.CTkFrame(self.results_frame)
        chart_frame.pack(fill="x", padx=10, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def batch_process(self):
        """Batch process folder"""
        folder_path = filedialog.askdirectory(title="Select Folder to Process")
        
        if not folder_path:
            return
        
        # Start batch processing
        thread = threading.Thread(target=self.run_batch_process, args=(folder_path,))
        thread.daemon = True
        thread.start()
    
    def run_batch_process(self, folder_path):
        """Run batch processing"""
        try:
            self.root.after(0, lambda: self.status_label.configure(text="Batch processing..."))
            
            # Find media files
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.mp4', '.avi', '.mov', '.mkv']
            files = []
            
            for ext in extensions:
                files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
            
            if not files:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No media files found"))
                return
            
            # Process files
            results = []
            for i, filename in enumerate(files):
                filepath = os.path.join(folder_path, filename)
                
                self.root.after(0, lambda i=i, total=len(files): 
                               self.status_label.configure(text=f"Processing {i+1}/{total}"))
                self.root.after(0, lambda i=i, total=len(files): 
                               self.progress_bar.set(i/total))
                
                result, confidence, _ = self.detector.detect_deepfake(filepath)
                results.append((filename, result, confidence))
            
            # Show summary
            self.root.after(0, lambda: self.show_batch_summary(results))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            self.root.after(0, lambda: self.status_label.configure(text="Ready"))
    
    def show_batch_summary(self, results):
        """Show batch processing summary"""
        total = len(results)
        real_count = sum(1 for _, result, _ in results if result == "REAL")
        fake_count = sum(1 for _, result, _ in results if result == "FAKE")
        uncertain_count = total - real_count - fake_count
        
        summary = f"""Batch Processing Complete!
        
Total files: {total}
✅ Real: {real_count} ({real_count/total:.1%})
⚠️ Fake: {fake_count} ({fake_count/total:.1%})
❓ Uncertain: {uncertain_count} ({uncertain_count/total:.1%})"""
        
        messagebox.showinfo("Batch Processing Summary", summary)
    
    def clear_results(self):
        """Clear all results"""
        self.current_file = None
        self.current_image = None
        self.current_video = None
        self.is_video = False
        
        # Release video capture if open
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        
        self.analysis_results = {}
        
        self.image_label.configure(image=None, text="No media selected")
        
        # Clear analysis tab
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        
        # Clear results tab
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.progress_bar.set(0)
        self.status_label.configure(text="Ready to detect deepfakes (images & videos)")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    app = ModernDeepfakeGUI()
    app.run()

if __name__ == "__main__":
    main()
