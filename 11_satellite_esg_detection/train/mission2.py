import os, cv2, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------------
# 1. YOLO로 bbox 추출 + crop
# -----------------------------
def crop_chimneys(yolo_model, img_dir, line_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for fname in tqdm(os.listdir(img_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(img_dir, fname)
        json_path = os.path.join(line_dir, fname.replace(".jpg", ".json"))
        if not os.path.exists(json_path):
            print(f"[WARN] Missing JSON for {fname}")
            continue

        # YOLO inference
        results = yolo_model(img_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]

        # JSON 라벨 읽기
        with open(json_path, encoding="utf8") as f:
            data = json.load(f)
        key = list(data.keys())[0]
        regions = data[key]["regions"]

        img = cv2.imread(img_path)
        for i, (box, reg) in enumerate(zip(boxes, regions)):
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            chi_height = float(reg["region_attributes"]["chi_height_m"])

            # crop 저장
            crop_name = f"{os.path.splitext(fname)[0]}_{i}.jpg"
            cv2.imwrite(os.path.join(out_dir, crop_name), crop)

            # 라벨 저장
            with open(os.path.join(out_dir, crop_name.replace(".jpg", ".txt")), "w") as ftxt:
                ftxt.write(str(chi_height))

    print(f"[INFO] Crops saved to {out_dir}")


# -----------------------------
# 2. Dataset 정의
# -----------------------------
class ChimneyHeightDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.samples = []
        self.transform = transform

        for fname in os.listdir(img_dir):
            if fname.endswith(".jpg"):
                label_path = os.path.join(img_dir, fname.replace(".jpg", ".txt"))
                if not os.path.exists(label_path):
                    continue
                with open(label_path) as f:
                    height = float(f.read().strip())
                self.samples.append((fname, height))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, height = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(height, dtype=torch.float32)


# -----------------------------
# 3. Transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


# -----------------------------
# 4. 학습 및 검증 함수
# -----------------------------
def train_regressor(train_loader, val_loader, epochs=20, lr=1e-4, device="cuda"):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for imgs, heights in train_loader:
            imgs, heights = imgs.to(device), heights.to(device)
            preds = model(imgs).squeeze()
            loss = criterion(preds, heights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Val
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, heights in val_loader:
                imgs, heights = imgs.to(device), heights.to(device)
                preds = model(imgs).squeeze()
                preds_all.extend(preds.cpu().numpy().reshape(-1))
                labels_all.extend(heights.cpu().numpy().reshape(-1))
        rmse = np.sqrt(np.mean((np.array(preds_all) - np.array(labels_all))**2))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss {total_loss/len(train_loader):.4f} | Val RMSE {rmse:.2f}")

    torch.save(model.state_dict(), "chimney_height_resnet.pth")
    print("[INFO] Model saved to chimney_height_resnet.pth")


# -----------------------------
# 5. 실행 예시
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ① YOLO로 bbox crop 생성
    yolo = YOLO("best.pt")
    crop_chimneys(yolo, "./data/train_data/KS", "./data/train_data/KS_LINE", "./cropped/train")
    crop_chimneys(yolo, "./data/val_data/VS_KS", "./data/val_data/VL_KS_LINE", "./cropped/val")

    # ② Dataset & DataLoader
    train_dataset = ChimneyHeightDataset("./cropped/train", transform)
    val_dataset   = ChimneyHeightDataset("./cropped/val", transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ③ 학습 실행
    train_regressor(train_loader, val_loader, epochs=50, lr=1e-4, device=device)
