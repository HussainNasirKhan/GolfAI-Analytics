import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List
import time
import ssl

# Handle SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

@dataclass
class SwingMetrics:
    spine_angle: float
    knee_flex: float
    hip_rotation: float
    shoulder_rotation: float
    arm_extension: float
    head_position: float
    weight_distribution: float
    phase: str

class GolfSwingAnalyzer:
    def __init__(self, video_path: str):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        )
        
        # Video source
        self.video_path = video_path
        
        # Initialize tracking
        self.session_start = time.time()
        self.posture_history = []

    def is_landmark_visible(self, landmark, visibility_threshold=0.5):
        """Check if a landmark is visible enough to be used."""
        return landmark.visibility > visibility_threshold

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def detect_swing_phase(self, landmarks):
        """Detect current phase of golf swing based on pose landmarks."""
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        shoulder_to_wrist = right_wrist.y - right_shoulder.y
        wrist_to_hip = right_wrist.x - right_hip.x
        
        if abs(shoulder_to_wrist) < 0.1 and abs(wrist_to_hip) < 0.1:
            return "ADDRESS"
        elif right_wrist.y < right_shoulder.y and right_wrist.x > right_hip.x:
            return "BACKSWING"
        elif right_wrist.y < right_shoulder.y and abs(wrist_to_hip) < 0.1:
            return "TOP"
        elif right_wrist.y > right_shoulder.y and right_wrist.x < right_hip.x:
            return "DOWNSWING"
        elif abs(shoulder_to_wrist) < 0.1 and right_wrist.x < right_hip.x:
            return "IMPACT"
        else:
            return "FOLLOW THROUGH"

    def calculate_metrics(self, landmarks):
        """Calculate swing metrics from landmarks."""
        # Initialize metrics
        spine_angle = 0
        knee_flex = 0
        hip_rotation = 0
        shoulder_rotation = 0
        arm_extension = 0
        head_position = 0
        weight_distribution = 50  # default centered

        # Calculate spine angle
        if all(self.is_landmark_visible(landmarks[i]) for i in [
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]):
            spine_angle = self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            )

        # Calculate knee flex
        if all(self.is_landmark_visible(landmarks[i]) for i in [
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]):
            knee_flex = self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            )

        # Calculate hip rotation using right side
        if all(self.is_landmark_visible(landmarks[i]) for i in [
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_KNEE
        ]):
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            center_x = right_hip.x - 0.2
            center_hip = type('Point', (), {'x': center_x, 'y': right_hip.y, 'z': right_hip.z})()
            hip_rotation = self.calculate_angle(
                center_hip,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            )

        # Calculate shoulder rotation
        if all(self.is_landmark_visible(landmarks[i]) for i in [
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW
        ]):
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            center_x = right_shoulder.x - 0.2
            center_shoulder = type('Point', (), {'x': center_x, 'y': right_shoulder.y, 'z': right_shoulder.z})()
            shoulder_rotation = self.calculate_angle(
                center_shoulder,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            )

        # Calculate arm extension
        if all(self.is_landmark_visible(landmarks[i]) for i in [
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]):
            arm_extension = self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            )

        # Calculate head position
        if all(self.is_landmark_visible(landmarks[i]) for i in [
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]):
            head_position = self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            )

        # Estimate weight distribution
        if self.is_landmark_visible(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]):
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            weight_shift = (right_ankle.x - right_hip.x) * 100
            weight_distribution = 50 + weight_shift

        phase = self.detect_swing_phase(landmarks)

        return SwingMetrics(
            spine_angle=spine_angle,
            knee_flex=knee_flex,
            hip_rotation=hip_rotation,
            shoulder_rotation=shoulder_rotation,
            arm_extension=arm_extension,
            head_position=head_position,
            weight_distribution=weight_distribution,
            phase=phase
        )

    def draw_text_with_background(self, image, text, position, font_scale=0.7,
                                thickness=2, text_color=(255, 255, 255),
                                bg_color=(0, 0, 0, 0.7), padding=10):
        """Draw text with background."""
        font = cv2.FONT_HERSHEY_DUPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background
        overlay = image.copy()
        cv2.rectangle(overlay,
                     (position[0] - padding, position[1] - text_height - padding),
                     (position[0] + text_width + padding, position[1] + padding),
                     bg_color[:3], -1)
        
        # Apply transparency
        cv2.addWeighted(overlay, bg_color[3], image, 1 - bg_color[3], 0, image)
        
        # Draw border
        cv2.rectangle(image,
                     (position[0] - padding, position[1] - text_height - padding),
                     (position[0] + text_width + padding, position[1] + padding),
                     (200, 200, 200), 1)
        
        # Draw text
        cv2.putText(image, text, position, font, font_scale, text_color, thickness)
        return text_height + 2 * padding

    def draw_feedback(self, image, metrics):
        """Draw metrics display without phase information."""
        # Text starting position
        x_pos = 20
        y_pos = 40
        
        # Draw metrics with individual backgrounds
        metrics_data = [
            ("SPINE ANGLE", f"{metrics.spine_angle:.1f}°"),
            ("KNEE FLEX", f"{metrics.knee_flex:.1f}°"),
            ("HIP ROTATION", f"{metrics.hip_rotation:.1f}°"),
            ("SHOULDER ROT", f"{metrics.shoulder_rotation:.1f}°"),
            ("ARM EXTENSION", f"{metrics.arm_extension:.1f}°")
        ]
        
        for label, value in metrics_data:
            y_pos += self.draw_text_with_background(
                image,
                f"{label}: {value}",
                (x_pos, y_pos),
                bg_color=(20, 20, 20, 0.75)
            )
            y_pos += 5  # Add small spacing between metrics

    def analyze(self):
        """Main analysis loop."""
        cap = cv2.VideoCapture(self.video_path)
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        output_path = self.video_path.rsplit('.', 1)[0] + '_analyzed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # Progress
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                print(f"\rAnalyzing video: {progress:.1f}%", end="")

                # Process frame
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)

                if results.pose_landmarks:
                    # Draw pose landmarks
                    self.mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    
                    # Calculate and draw metrics
                    metrics = self.calculate_metrics(results.pose_landmarks.landmark)
                    self.draw_feedback(image, metrics)

                # Save and display frame
                out.write(image)
                cv2.imshow('Golf Swing Analysis', image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print("\nAnalysis complete!")
            print(f"Analyzed video saved as: {output_path}")
            cap.release()
            out.release()
            cv2.destroyAllWindows()

def main():
    # Initialize and run analyzer
    video_path = "golf_swing.mp4"  # Replace with your video path
    analyzer = GolfSwingAnalyzer(video_path)
    analyzer.analyze()

if __name__ == "__main__":
    main()