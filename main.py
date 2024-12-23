import numpy as np
import requests
import cv2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

#for loading pre-trained model
model = InceptionV3(weights='imagenet')

def classify_image(image):
#pre-processing the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (299, 299))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    predictions = model.predict(image_array)
    top_prediction = decode_predictions(predictions, top=1)[0][0]
    label = top_prediction[1]
    return label

#for fetching the recipes
def fetch_recipes(ingredient):
    api_key = "your API key"
    base_url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": ingredient,
        "number": 5, #this specify the number of recipes to find
        "cuisine": "Indian", #this is an optional filter specifically for Indian cuisine 
        "apiKey": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching recipes: {response.status_code} - {response.text}")
        return []

#to display the details of the recipe
def display_recipes(recipes):
    if not recipes:
        print("No recipes found.")
        return
    print("\nRecipes suggested based on the identified ingredient:")
    for recipe in recipes:
        recipe_title = recipe['title'].replace(' ', '-').lower()
        recipe_id = recipe['id']
        recipe_link = f"https://spoonacular.com/recipes/{recipe_title}-{recipe_id}"
        print(f"- {recipe['title']}")
        print(f"  Used Ingredients: {[item['name'] for item in recipe.get('usedIngredients', [])]}")
        print(f"  Missed Ingredients: {[item['name'] for item in recipe.get('missedIngredients', [])]}")
        print(f"  Recipe Link: {recipe_link}\n")

#Now, the function to capture image from webcam
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()
    print("Press 'c' to capture an image, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Capturing image...")
            label = classify_image(frame)
            print(f"The object in the image is: {label}")
            print(f"Fetching recipes for {label}...")
            recipes = fetch_recipes(label)
            display_recipes(recipes)
            break
        elif key == ord('q'): #quit when 'q' is pressed
            print("Exiting...")
            break
    #release webcam and openCV window
    cap.release()
    cv2.destroyAllWindows()
