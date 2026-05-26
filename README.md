# Audio Matcher (Shazam Replica)

A Python-based digital signal processing tool that scans a long audio file to find the exact hidden location of a short audio clip. 

## 💡 The Problem & Solution
* **The Problem:** You can't easily match audio by just looking at raw sound waves, because background noise or tiny changes ruin the comparison.
* **The Solution:** This app converts audio into frequency data (how high or low the pitch is) so it can accurately match the unique "sound signature" of the clip, just like Shazam does.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** NumPy (data math), SciPy (audio file handling), Matplotlib (plotting results)

## ⚙️ How It Works
1. **Load & Clean:** Reads a `.wav` audio file, converts it to mono, and prepares the data.
2. **Analyze Clip:** Extracts a short audio snippet and calculates its frequency footprint.
3. **Scan:** Slides a search window across the main audio track.
4. **Score & Locate:** Compares the frequencies at every step using a vector similarity formula, finding the exact timestamp where the match is highest.

## 📊 Project Outputs
The script automatically prints the matching timestamps and generates easy-to-read charts showing:
* The original audio waveform vs. the found segment.
* A timeline tracking the algorithm's confidence score over time to prove the match is accurate.
