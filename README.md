
---

## ⚡ Quick Start

### 1. **Clone the repository**

```bash
git clone https://github.com/sdwc-dev/2d-floorplan-to-3d.git
cd 2d-floorplan-to-3d
```

### 2. **Install dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Run the app**

```bash
streamlit run app.py
```

### 4. **Open in your browser**

Go to [http://localhost:8501](http://localhost:8501)




---

## ⚠️ Blender Dependency

- For advanced 3D boolean operations (cutting wall openings for doors/windows), the app tries to use [Blender](https://www.blender.org/).
- **If Blender is not available** (e.g., on Streamlit Cloud), the app will use a slower fallback engine and show a warning. The app will still work, but some 3D features may be less accurate for complex floorplans.

---

## 📝 Notes

- Make sure `best.pt`, `door.obj`, and `window.obj` are present in your project directory.
- The app supports both detection-based and rule-based wall generation.
- You can customize colors and download the generated 3D models.

---


