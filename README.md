# facerecogdaa
# Face Similarity Search Engine
Using Dynamic Programming (DP) and Longest Common Subsequence (LCS)
# Project Description
This project implements a Face Similarity Search Engine using OpenCV for face detection and Dynamic Programming–based Longest Common Subsequence (LCS) algorithm for similarity matching.
Instead of traditional machine learning or deep learning models, this system focuses on an algorithmic approach where facial features are represented as sequences and compared using DP + LCS to determine similarity between faces.
The face with the highest LCS similarity score is returned as the closest match.
# Working Methodology
1️⃣ Face Detection
Convert image to grayscale
Detect face using Haar Cascade Classifier
Crop the detected face (ROI)
2️⃣ Feature Extraction
Resize face to a fixed dimension
Extract pixel intensity / edge-based values
Convert extracted values into a 1D sequence
Face A → [122, 119, 121, 130, 128]
Face B → [120, 119, 121, 129, 127]
3️⃣ Dynamic Programming + LCS
LCS is computed using DP table
Each stored face sequence is compared with input sequence
LCS length indicates similarity strength
if X[i] == Y[j]:
    dp[i][j] = 1 + dp[i-1][j-1]
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
4️⃣ Similarity Search
Compute LCS score with all dataset faces
Normalize similarity score
Rank faces based on similarity
Display top match or label as Unknown
# ⏱️ Time & Space Complexity
LCS Time Complexity: O(n × m)
Space Complexity: O(n × m)
Where n and m are lengths of face feature sequences