import sys
import cv2
import torch

sys.path.append('yolov5')

from models.experimental import attempt_load

model = attempt_load('detectModel/weights/best.pt')

model.eval()

with open('classes/class.names', 'r') as f:
    class_names = f.read().strip().split('\n')
print(class_names)

def detect(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))

    img_tensor = torch.from_numpy(img).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        results = model(img_tensor)

    detections = results[0]

    print("Структура detections:")
    print(detections)

    for detection in detections:
        if len(detection) > 4:
            box = detection[:4]

            print(f"Координаты бокса: {box}")
            print(f"Детекция: {detection}")

            # Возможно, уверенность и класс расположены в другом месте
            conf = detection[4].item() if detection[4].numel() == 1 else detection[4][0].item()
            cls_probs = detection[5:]

            cls = torch.argmax(
                cls_probs).item() if cls_probs.numel() > 0 else -1
            if cls >= 0 and cls < len(model.names):
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(img, label, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif cls >= 115866 and cls <= 115869:
                # print(f"Класс {class_names[cls - 115866]}")
                return class_names[cls - 115866]
            else:
                print(f"Класс {cls} не найден в model.names")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detections', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <body>
            <form action="/uploadfile/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" />
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    with open(f"./uploaded_files/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())
    return {"filename": detect(f"./uploaded_files/{file.filename}")}

