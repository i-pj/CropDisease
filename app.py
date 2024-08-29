import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("./best.pt")

st.title("Crop Disease Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Run Model"):
    if uploaded_file is not None:
        image = cv2.imdecode(
            np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
        )
        results = model(image)
        # boxes
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy()
                cv2.rectangle(
                    image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
                cv2.putText(
                    image,
                    f"{result.boxes.cls[0].cpu().numpy().item():.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            st.write(
                f"Image 1/1 {uploaded_file.name}: {image.shape[1]}x{image.shape[0]}"
            )
            for i, (cls, conf) in enumerate(
                zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy())
            ):
                st.write(f"{i+1}. {model.names[int(cls)]}, Confidence: {conf:.2f}")
            st.write(
                f"Speed: {result.speed['preprocess']:.2f}ms preprocess, {result.speed['inference']:.2f}ms inference, {result.speed['postprocess']:.2f}ms postprocess per image"
            )
            st.write(
                f"**Image Class:** {model.names[int(result.boxes.cls[0].cpu().numpy().item())]}"
            )

        st.image(image, caption="Object Detection Results")
    else:
        st.error("Please upload an image")
