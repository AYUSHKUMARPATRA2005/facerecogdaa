from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Extract facial embedding using DeepFace (Facenet model by default)
def extract_features(image_path):
    embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=True)
    return embedding[0]["embedding"]

# Compute cosine similarity between embeddings
def cosine_sim(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Quantize embeddings by rounding to integers for LCS
def quantize_embedding(embedding):
    # You can customize quantization granularity here
    return [int(round(x * 100)) for x in embedding]

# Compute LCS length using DP for two sequences
def lcs_length(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# Calculate LCS similarity normalized by average length
def lcs_similarity(seq1, seq2):
    lcs_len = lcs_length(seq1, seq2)
    avg_len = (len(seq1) + len(seq2)) / 2
    return lcs_len / avg_len if avg_len > 0 else 0

# Combined similarity: weighted sum of cosine similarity and LCS similarity
def combined_similarity(embedding1, embedding2, alpha=0.7):
    cosine_score = cosine_sim(embedding1, embedding2)
    seq1 = quantize_embedding(embedding1)
    seq2 = quantize_embedding(embedding2)
    lcs_score = lcs_similarity(seq1, seq2)
    return alpha * cosine_score + (1 - alpha) * lcs_score

# Face similarity search function with combined similarity
def face_similarity_search(query_img_path, database_img_paths):
    query_feature = extract_features(query_img_path)

    scores = []
    for img_path in database_img_paths:
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} does not exist!")
            continue
        db_feature = extract_features(img_path)
        sim_score = combined_similarity(query_feature, db_feature)
        scores.append((img_path, sim_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# Example usage
if __name__ == '__main__':
    query_image = 'ayush.jpg'
    database_images = ['ayush.jpg', 'rohan.jpg','swatikant-sir.jpg']

    results = face_similarity_search(query_image, database_images)
    for img_path, score in results:
        print(f"{img_path} similarity score: {score * 100:.2f}%")



# import cv2
# from deepface import DeepFace
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Pre-extract embeddings for database images once
# def prepare_database(database_img_paths):
#     database_embeddings = []
#     for img_path in database_img_paths:
#         embedding = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=True)
#         database_embeddings.append((img_path, embedding[0]['embedding']))
#     return database_embeddings

# def cosine_sim(vec1, vec2):
#     vec1 = np.array(vec1).reshape(1, -1)
#     vec2 = np.array(vec2).reshape(1, -1)
#     return cosine_similarity(vec1, vec2)[0][0]

# def real_time_face_similarity(database_embeddings):
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         try:
#             # Pass frame directly (NO saving)
#             query_embedding = DeepFace.represent(
#                 img_path=frame,
#                 model_name='Facenet',
#                 enforce_detection=True
#             )[0]['embedding']

#             scores = []
#             for img_path, db_embedding in database_embeddings:
#                 score = cosine_sim(query_embedding, db_embedding)
#                 scores.append((img_path, score))

#             best_match = max(scores, key=lambda x: x[1])

#             cv2.putText(
#                 frame,
#                 f"Best match: {best_match[0].split('/')[-1]} {best_match[1]*100:.2f}%",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 255, 0),
#                 2
#             )

#         except Exception:
#             cv2.putText(
#                 frame,
#                 "No face detected",
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 0, 255),
#                 2
#             )

#         cv2.imshow('Real-time Face Similarity', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     database_images = ['ayush.jpg', 'rohan.jpg', 'swatikant-sir.jpg']
#     database_embeddings = prepare_database(database_images)
#     real_time_face_similarity(database_embeddings)



# import cv2
# from deepface import DeepFace
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Pre-extract embeddings for database images once
# def prepare_database(database_img_paths):
#     database_embeddings = []
#     for img_path in database_img_paths:
#         embedding = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=True)
#         database_embeddings.append((img_path, embedding[0]['embedding']))
#     return database_embeddings

# def cosine_sim(vec1, vec2):
#     vec1 = np.array(vec1).reshape(1, -1)
#     vec2 = np.array(vec2).reshape(1, -1)
#     return cosine_similarity(vec1, vec2)[0][0]

# def real_time_face_similarity(database_embeddings):
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Save current frame temporarily to pass to DeepFace
#         cv2.imwrite('temp_frame.jpg', frame)
#         try:
#             # Extract embedding for face in current frame
#             query_embedding = DeepFace.represent(img_path='temp_frame.jpg', model_name='Facenet', enforce_detection=True)[0]['embedding']

#             scores = []
#             for img_path, db_embedding in database_embeddings:
#                 score = cosine_sim(query_embedding, db_embedding)
#                 scores.append((img_path, score))

#             # Find best match
#             best_match = max(scores, key=lambda x: x[1])

#             # Display best match and similarity on frame
#             cv2.putText(frame, f"Best match: {best_match[0].split('/')[-1]} {best_match[1]*100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         except Exception as e:
#             # Face not detected
#             cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         cv2.imshow('Real-time Face Similarity', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     database_images = ['ayush.jpg', 'rohan.jpg','swatikant-sir.jpg']
#     database_embeddings = prepare_database(database_images)
#     real_time_face_similarity(database_embeddings)





# import cv2
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Load and preprocess image
# def preprocess_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (100, 100))  # Resize for consistent shape
#     return img

# # Extract feature vector using simple descriptor (pixel intensities flattened here)
# def extract_features(img):
#     return img.flatten()

# # Compute similarity using cosine similarity
# def cosine_sim(vec1, vec2):
#     vec1 = vec1.reshape(1, -1)
#     vec2 = vec2.reshape(1, -1)
#     return cosine_similarity(vec1, vec2)[0][0]

# # Dynamic Programming + LCS for subsequence similarity (for strings, can extend to features)
# def lcs(X, Y):
#     m = len(X)
#     n = len(Y)
#     L = [[0] * (n+1) for i in range(m+1)]
#     for i in range(m+1):
#         for j in range(n+1):
#             if i == 0 or j == 0:
#                 L[i][j] = 0
#             elif X[i-1] == Y[j-1]:
#                 L[i][j] = L[i-1][j-1] + 1
#             else:
#                 L[i][j] = max(L[i-1][j], L[i][j-1])
#     return L[m][n]

# # Face similarity search engine main function
# def face_similarity_search(query_img_path, database_img_paths):
#     query_img = preprocess_image(query_img_path)
#     query_feature = extract_features(query_img)

#     scores = []
#     for img_path in database_img_paths:
#         db_img = preprocess_image(img_path)
#         db_feature = extract_features(db_img)

#         # Compute similarity by cosine sim of vectors
#         sim_score = cosine_sim(query_feature, db_feature)

#         scores.append((img_path, sim_score))

#     # Sort images by descending similarity score
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return scores

# # Example usage
# if __name__ == '__main__':
#     query_image = 'ayush.jpg'
#     database_images = ['ayush.jpg', 'rohan.jpg','new_img - Copy.jpg']

#     results = face_similarity_search(query_image, database_images)
#     for img_path, score in results:
#         percentage = score * 100
#         print(f"{img_path} similarity score: {percentage}")









# from deepface import DeepFace
# from sklearn.metrics.pairwise import cosine_similarity
# import os
# import numpy as np


# # Extract facial embedding using DeepFace (Facenet model by default)
# def extract_features(image_path):
#     embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=True)
#     return embedding[0]["embedding"]

# # Compute cosine similarity between embeddings
# def cosine_sim(vec1, vec2):
#     vec1 = np.array(vec1).reshape(1, -1)
#     vec2 = np.array(vec2).reshape(1, -1)
#     return cosine_similarity(vec1, vec2)[0][0]

# # Face similarity search function with embeddings and percentage display
# def face_similarity_search(query_img_path, database_img_paths):
#     query_feature = extract_features(query_img_path)

#     scores = []
#     for img_path in database_img_paths:
#         if not os.path.exists(img_path):
#             print(f"Warning: {img_path} does not exist!")
#             continue
#         db_feature = extract_features(img_path)
#         sim_score = cosine_sim(query_feature, db_feature)
#         scores.append((img_path, sim_score))

#     scores.sort(key=lambda x: x[1], reverse=True)
#     return scores

# # Example usage
# if __name__ == '__main__':
#     query_image = 'ayush.jpg'
#     database_images = ['ayush.jpg', 'rohan.jpg']

#     results = face_similarity_search(query_image, database_images)
#     for img_path, score in results:
#         print(f"{img_path} similarity score: {score * 100:.2f}%")




