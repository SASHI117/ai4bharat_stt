# ✅ AI4Bharat ASR — Local Setup
---

## 🧱 System Requirements (Important)

* **Python 3.9.x** ✅ (DO NOT use 3.11 / 3.12 / 3.13)

---

## 1️⃣ Install Python (CRITICAL)

Download **Python 3.9.x** from:
👉 [https://www.python.org/downloads/release/python-3913/](https://www.python.org/downloads/release/python-3913/)

During installation:
✔️ **Check “Add Python to PATH”**

Verify:

```powershell
python --version
# Python 3.9.x
```

---

## 2️⃣ Create Virtual Environment (venv)

From your project folder:

```powershell
cd ai4bharat_asr
python -m venv venv
```

Activate:

```powershell
venv\Scripts\activate
```

You should see:

```
(venv)
```

---

## 3️⃣ Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

## 4️⃣ Install EXACT dependencies


```powershell
pip install -r requirements.txt
```

---

## 6️⃣ Hugging Face Login (Once)

```powershell
huggingface-cli login
```

Paste your token
(make sure you already requested access to the model)

---

## 7️⃣ Model ID (Correct One)

✅ Use ONLY:

```python
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
```

---

## 8️⃣ Run Transcription

```powershell
python transcribe_new.py
```
