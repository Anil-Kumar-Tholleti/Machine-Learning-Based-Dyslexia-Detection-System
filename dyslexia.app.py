import streamlit as st
from PIL import Image
import os
from textblob import TextBlob
import language_tool_python
import requests
import pandas as pd
import random
import speech_recognition as sr
import pyttsx3
import time
import eng_to_ipa as ipa
import pytesseract
import cv2
import uuid
import easyocr
from streamlit_drawable_canvas import st_canvas
import numpy as np
import threading
import matplotlib.pyplot as plt
is_speaking = False  # Global flag to track speech status
speech_thread = None  # Store the speech thread


# from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
# from msrest.authentication import CognitiveServicesCredentials

import time

from abydos.phonetic import Soundex, Metaphone, Caverphone, NYSIIS

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
def talk(word: str):
    engine = pyttsx3.init()
    engine.say(word)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Listening... Please speak now")
            audio_data = r.listen(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Speech not recognized."
    except sr.RequestError:
        return "Speech recognition service unavailable."
# Define words for practice and phonetics quiz
words_to_practice = ["schedule", "genre", "subtle", "debt", "queue", "liaison", "paradigm", "bourgeois"] * 6  # 50 words
quiz_words = {"schedule": "skɛdʒuːl", "genre": "ʒɒ̃rə", "subtle": "ˈsʌtəl", "liaison": "liˈeɪ.zɒn", "paradigm": "ˈpær.ə.daɪm", "bourgeois": "ˈbʊərʒ.wɑː"}
def calculate_accuracy(word1, word2):
    max_length = max(len(word1), len(word2))
    if max_length == 0:
        return 100  # If both are empty, consider it perfect
    levenshtein_dist = levenshtein(word1, word2)
    accuracy = max(0, (1 - levenshtein_dist / max_length) * 100)  # Ensure it's never negative
    return round(accuracy, 2)


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Convert the image to grayscale and apply thresholding"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_text_easyocr(image_path):
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader for English
    result = reader.readtext(image_path)  # Extract text from image
    extracted_text = " ".join([res[1] for res in result])  # Combine extracted words
    return extracted_text

def image_to_text(path):
    try:
        reader = easyocr.Reader(['en'])  # Initialize OCR reader for English
        result = reader.readtext(path)

        extracted_text = [text[1] for text in result]  # Extract detected text
        return " ".join(extracted_text) if extracted_text else "No text detected."

    except Exception as e:
        return f"Error: {str(e)}"
# # image to text API authentication
# subscription_key_imagetotext = "1780f5636509411da43040b70b5d2e22"
# endpoint_imagetotext = "https://prana-------------v.cognitiveservices.azure.com/"
# computervision_client = ComputerVisionClient(
#     endpoint_imagetotext, CognitiveServicesCredentials(subscription_key_imagetotext))

# # '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# # text correction API authentication
# api_key_textcorrection = "7aba4995897b4dcaa86c34ddb82a1ecf"
# endpoint_textcorrection = "https://api.bing.microsoft.com/v7.0/SpellCheck"

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

#my_tool = language_tool_python.LanguageTool('en-US')

# Set the path to the manually downloaded LanguageTool JAR
from textblob import TextBlob
import language_tool_python

my_tool = language_tool_python.LanguageToolPublicAPI('en-US')










# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# method for extracting the text


def image_to_text(path):
    read_image = open(path, "rb")
    read_response = computervision_client.read_in_stream(read_image, raw=True)
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(5)

    text = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                text.append(line.text)

    return " ".join(text)

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# method for finding the spelling accuracy


def spelling_accuracy(extracted_text):
    spell_corrected = TextBlob(extracted_text).correct()
    return ((len(extracted_text) - (levenshtein(extracted_text, spell_corrected)))/(len(extracted_text)+1))*100

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# method for gramatical accuracy


def gramatical_accuracy(extracted_text):
    spell_corrected = TextBlob(extracted_text).correct()
    correct_text = my_tool.correct(spell_corrected)
    extracted_text_set = set(spell_corrected.split(" "))
    correct_text_set = set(correct_text.split(" "))
    n = max(len(extracted_text_set - correct_text_set),
            len(correct_text_set - extracted_text_set))
    return ((len(spell_corrected) - n)/(len(spell_corrected)+1))*100

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# percentage of corrections


def percentage_of_corrections(text):
    corrected_text = TextBlob(text).correct()
    grammar_checked = my_tool.correct(str(corrected_text))
    
    original_words = text.split()
    corrected_words = grammar_checked.split()
    
    return (len(set(original_words) - set(corrected_words)) / len(original_words)) * 100


# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# percentage of phonetic accuracy


def percentage_of_phonetic_accuraccy(extracted_text: str):
    soundex = Soundex()
    metaphone = Metaphone()
    caverphone = Caverphone()
    nysiis = NYSIIS()
    spell_corrected = TextBlob(extracted_text).correct()

    extracted_text_list = extracted_text.split(" ")
    extracted_phonetics_soundex = [soundex.encode(
        string) for string in extracted_text_list]
    extracted_phonetics_metaphone = [metaphone.encode(
        string) for string in extracted_text_list]
    extracted_phonetics_caverphone = [caverphone.encode(
        string) for string in extracted_text_list]
    extracted_phonetics_nysiis = [nysiis.encode(
        string) for string in extracted_text_list]

    extracted_soundex_string = " ".join(extracted_phonetics_soundex)
    extracted_metaphone_string = " ".join(extracted_phonetics_metaphone)
    extracted_caverphone_string = " ".join(extracted_phonetics_caverphone)
    extracted_nysiis_string = " ".join(extracted_phonetics_nysiis)

    spell_corrected_list = spell_corrected.split(" ")
    spell_corrected_phonetics_soundex = [
        soundex.encode(string) for string in spell_corrected_list]
    spell_corrected_phonetics_metaphone = [
        metaphone.encode(string) for string in spell_corrected_list]
    spell_corrected_phonetics_caverphone = [
        caverphone.encode(string) for string in spell_corrected_list]
    spell_corrected_phonetics_nysiis = [nysiis.encode(
        string) for string in spell_corrected_list]

    spell_corrected_soundex_string = " ".join(
        spell_corrected_phonetics_soundex)
    spell_corrected_metaphone_string = " ".join(
        spell_corrected_phonetics_metaphone)
    spell_corrected_caverphone_string = " ".join(
        spell_corrected_phonetics_caverphone)
    spell_corrected_nysiis_string = " ".join(spell_corrected_phonetics_nysiis)

    soundex_score = (len(extracted_soundex_string)-(levenshtein(extracted_soundex_string,
                     spell_corrected_soundex_string)))/(len(extracted_soundex_string)+1)
    # print(spell_corrected_soundex_string)
    # print(extracted_soundex_string)
    # print(soundex_score)
    metaphone_score = (len(extracted_metaphone_string)-(levenshtein(extracted_metaphone_string,
                       spell_corrected_metaphone_string)))/(len(extracted_metaphone_string)+1)
    # print(metaphone_score)
    caverphone_score = (len(extracted_caverphone_string)-(levenshtein(extracted_caverphone_string,
                        spell_corrected_caverphone_string)))/(len(extracted_caverphone_string)+1)
    # print(caverphone_score)
    nysiis_score = (len(extracted_nysiis_string)-(levenshtein(extracted_nysiis_string,
                    spell_corrected_nysiis_string)))/(len(extracted_nysiis_string)+1)
    # print(nysiis_score)
    return ((0.5*caverphone_score + 0.2*soundex_score + 0.2*metaphone_score + 0.1 * nysiis_score))*100

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def get_feature_array(path: str):
    feature_array = []
    extracted_text = image_to_text(path)
    feature_array.append(spelling_accuracy(extracted_text))
    feature_array.append(gramatical_accuracy(extracted_text))
    feature_array.append(percentage_of_corrections(extracted_text))
    feature_array.append(percentage_of_phonetic_accuraccy(extracted_text))
    return feature_array

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def generate_csv(folder: str, label: int, csv_name: str):
    arr = []
    for image in os.listdir(folder):
        path = os.path.join(folder, image)
        feature_array = get_feature_array(path)
        feature_array.append(label)
        # print(feature_array)
        arr.append(feature_array)
        print(feature_array)
    print(arr)
    pd.DataFrame(arr, columns=["spelling_accuracy", "gramatical_accuracy", " percentage_of_corrections",
                 "percentage_of_phonetic_accuraccy", "presence_of_dyslexia"]).to_csv("test1.csv")

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


def score(input):
    if input[0] <= 96.40350723266602:
        var0 = [0.0, 1.0]
    else:
        if input[1] <= 99.1046028137207:
            var0 = [0.0, 1.0]
        else:
            if input[2] <= 2.408450722694397:
                if input[2] <= 1.7936508059501648:
                    var0 = [1.0, 0.0]
                else:
                    var0 = [0.0, 1.0]
            else:
                var0 = [1.0, 0.0]
    return var0

# '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

# deploying the model


st.set_page_config(page_title="Dyslexia Webapp")

hide_menu_style = """
<style>
#MainMenu {visibility: hidden; }
footer {visibility: hidden; }
</style>
"""


st.markdown(hide_menu_style, unsafe_allow_html=True)
st.header("Dyslexia Web APP")

# tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Writing", "Pronunciation", "Dictation", "About"])
tabs = ["Home", "Writing", "Pronunciation", "Dictation", "Practices", "About"]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs)

with tab1:
    st.header("Home Page")
    st.write("""
    Dyslexia is a learning disorder that involves difficulty reading due to problems identifying 
    speech sounds and learning how they relate to letters and words (decoding). Also called a 
    reading disability, dyslexia is a result of individual differences in areas of the brain that 
    process language.

Dyslexia is not due to problems with intelligence, hearing or vision. Most children with dyslexia 
can succeed in school with tutoring or a specialized education program. Emotional support also plays 
an important role.

Though there's no cure for dyslexia, early assessment and intervention result in the best outcome. 
Sometimes dyslexia goes undiagnosed for years and isn't recognized until adulthood, but it's never 
too late to seek help.""")

    img1 = Image.open(r"C:\Users\tholl\OneDrive\Pictures\OneDrive\Documents\Anil reddy\charvi\images\img1.jpg")
    st.image(img1)

    st.subheader("Dyslexia- India")
    st.write("""
With regard to sociodemographic variables of primary school students, majority of the students 
56 (56%) belong to the age group of 6 years and 44 (44%) were 7 years. On gender, 57 (57%) were 
female and 43 (43%) were male. With regard to the religion, 88 (88%) were Hindu, 8 (8%) were 
Muslims, and 4 (4%) were Christians. With respect to occupational status of father, majority were 
private employee (47%), daily wages 39%, government employee 10%, and business 4%. Regarding the 
occupational status of mother, most of them were housewife (75%), daily worker 15%, private employee 9%, and government employee 1%.

Among the 100 samples, 50% were selected from I standard and another 50% were selected from II 
standard. With respect to the place of residence, 51 (51%) are from urban area and 49 (49%) are 
from rural area. In terms of language spoken by them Majority of the primary school students 95 
(95%) of them were speaking Kannada commonly at home and 05 (05%) of them were speaking Telugu at 
home. The entire primary school students, i.e., 100 (100%) of them, are speaking English at school. 
In connection with the data on their academic performance, 50 (50%) are having average academic performance, 44 (44%) are having good, 
and 6 (6%) are having excellent academic performance.""")


with tab2:
    st.title("Dyslexia Detection Using Handwriting Samples")
    st.write("This is a simple web app that works based on machine learning techniques. This application can predict the presence of dyslexia from the handwriting sample of a person.")

    image = st.file_uploader("Upload the handwriting sample that you want to test", type=["jpg", "png", "jpeg"])
    
    if image is not None:
        st.write("Please review the image selected")
        st.image(image, width=224)
        
        # Save the uploaded image
        with open("temp.jpg", "wb") as f:
            f.write(image.getbuffer())

    if st.button("Predict", help="Click after uploading the correct image"):
        try:
            # Extract text from the image
            extracted_text = extract_text_easyocr("temp.jpg")
            st.write("Extracted Text:", extracted_text)

            # Apply your Dyslexia prediction model on extracted_text
            # Example (replace with your actual ML model)
            if len(extracted_text.split()) < 10:  # Example rule: Short text might indicate issues
                st.write("High chance of dyslexia or dysgraphia")
            else:
                st.write("Low chance of dyslexia or dysgraphia")

        except Exception as e:
            st.write("Something went wrong:", str(e))

with tab3:

#'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

    def get_10_word_array(level: int):
        if (level == 1):
            voc = pd.read_csv("data\intermediate_voc.csv")
            arr = voc.squeeze().to_numpy()
            selected_list = random.sample(list(arr), 10)
            return selected_list
        elif(level == 2):
            voc = pd.read_csv("data\intermediate_voc.csv")
            # return (type(voc))
            arr = voc.squeeze().to_numpy()
            selected_list = random.sample(list(arr), 10) 
            return selected_list
        else:
            return ([])
    
#'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
    
    def listen_for(seconds: int):
        print("Listening function started...")  # Debugging
        r = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
                print("Listening for speech...")
                audio_data = r.record(source, duration=seconds)
                
                try:
                    text = r.recognize_google(audio_data)
                    print(f"Recognized: {text}")
                    return text
                except sr.UnknownValueError:
                    return "❌ Speech not recognized. Please speak clearly."
                except sr.RequestError:
                    return "⚠️ Speech recognition service unavailable."

        except OSError:
            return "⚠️ No microphone found. Please connect a microphone."

 
 #'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
    
    def talk(Word : str):
        engine = pyttsx3.init()
        engine.say(Word)
        engine.runAndWait()

    
    # def talk(Word: str):
    #     global is_speaking
    #     if not is_speaking:
    #         return  # Stop immediately if the flag is turned off
    #     engine = pyttsx3.init()
    #     engine.say(Word)
    #     engine.runAndWait()

#'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
    
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1       # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

#'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

    def check_pronounciation(str1, str2):
        if not str2:
            return len(str1), 0  # Max inaccuracy (100%) if no speech

        s1_words = str1.split()
        s2_words = str2.split()
        
        correct_words = 0
        total_words = len(s1_words)

        for w1, w2 in zip(s1_words, s2_words):
            if levenshtein(w1, w2) == 0:
                correct_words += 1

        accuracy = (correct_words / total_words) * 100
        inaccuracy = (1 - (correct_words / total_words)) * 100

        return inaccuracy, accuracy


#'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
    
    # def dictate_10_words(level : int):
    #     words = get_10_word_array(level)
    #     for i in words:
    #         talk(i)
    #         time.sleep(8)
    #     return words
    def dictate_10_words(level: int):
        global is_speaking
        is_speaking = True  # Set flag to start speaking

        words = get_10_word_array(level)
        
        def speech_task():
            for i in words:
                if not is_speaking:
                    break  # Stop speech if tab is changed
                talk(i)
                time.sleep(2)  # Adjust time delay if needed

        global speech_thread
        speech_thread = threading.Thread(target=speech_task)
        speech_thread.start()

        return words


#'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

    def random_seq():
        list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
        return " ".join(random.sample(list, 5))

#'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''



#'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

    tab1, tab2, tab3 = st.tabs(["Home", "pronounciation test", "phonetics"])

    level = 1


    with tab1:
        st.title("A Test for Dyslexia")
        option = st.selectbox(
            "select your standard", ('2nd-4th', '5th-7th'), key= "pro")
        if option=='2nd-4th':
            level = 2
        elif option == '5th-7th':
            level = 1

    with tab2:
        st.header("The pronounciation and reading ability of the user will be measured here")
        pronounciation_test = st.button("Start a pronouncation test")
        pronounciation_inaccuracy = 0
        
        if pronounciation_test:
            st.subheader("Please repeate the following words you only has 10 seconds to do that.")
         
            arr = get_10_word_array(level)
            for i in range(len(arr)):
                arr[i] = str(arr[i])
                arr[i] = arr[i].strip()

            str_displayed = str(" ".join(arr))
            words = st.text(">> " + "\n>>".join(arr) )
            status = st.text("listenning........")
            str_pronounced = listen_for(10)
            status.write("Time up! calculating inacuracy......")
        
        
            pronounciation_inaccuracy, pronounciation_accuracy = check_pronounciation(str_displayed, str_pronounced)

        
            words.write("the pronounciation inacuuracy is: " + str(pronounciation_inaccuracy))
            status.write("original : " + ipa.convert(str_displayed) )
            st.write("\npronounced: " + ipa.convert(str_pronounced))
            st.write(f"✅ Pronunciation Accuracy: {pronounciation_accuracy:.2f}%")
            st.progress(int(pronounciation_accuracy))  # Show a progress bar

            # Display a colored box based on accuracy level
            if pronounciation_accuracy > 80:
                st.success("🌟 Excellent Pronunciation!")
            elif pronounciation_accuracy > 50:
                st.warning("⚠️ Needs Improvement!")
            else:
                st.error("❌ Poor Pronunciation, Keep Practicing!")

            st.write(f"❌ Pronunciation Inaccuracy: {pronounciation_inaccuracy:.2f}%")


    with tab3:
        st.title("🧠 Dyslexia Practice & Improvement")
        st.write("This section helps users practice pronunciation, spelling, reading, and writing with interactive exercises.")
        
        # 🎤 Pronunciation & Speech Practice
        st.header("🎤 Pronunciation & Speech Practice")
        selected_word = st.selectbox("Choose a word to practice:", words_to_practice, key="word_practice")

        if st.button("🔊 Hear Pronunciation", key="hear_pronunciation"):
            try:
                talk(selected_word)  # Ensure talk() function is correctly implemented
            except Exception as e:
                st.error(f"Error playing pronunciation: {e}")

        if st.button("🎙️ Try Pronouncing It", key="try_pronounce"):
            user_pronounced = listen()
            if user_pronounced:
                st.write(f"🗣️ You said: **{user_pronounced}**")
                accuracy = calculate_accuracy(selected_word, user_pronounced)
                st.write(f"✅ Accuracy: {accuracy}%")
                st.progress(min(max(accuracy / 100, 0), 1))  
            else:
                st.warning("⚠️ Please try again.")

        # 📖 Phonetics Quiz
        st.header("📖 Phonetics Quiz")
        st.markdown("### 🎯 Phonetics Quiz")
        
        # Expanded Quiz Words with Correct Pronunciations
        quiz_words = {
            "schedule": ["skɛdʒuːl", "ʃɛdjuːl", "skɪdʒuːl"],
            "genre": ["ʒɒ̃rə", "ʒɑːnrə", "gɛnre"],
            "subtle": ["ˈsʌtəl", "ˈsuːbtəl", "sʌbtl"],
            "liaison": ["liˈeɪ.zɒn", "leɪ.zɒn", "laɪ.zɒn"],
            "paradigm": ["ˈpær.ə.daɪm", "ˈpær.ə.dɪg.m", "pær.ə.daɪ.ʤəm"],
            "bourgeois": ["ˈbʊərʒ.wɑː", "ˈbuːɡ.wɑː", "bʊrˈɡeɪ"],
            "colonel": ["ˈkɝː.nəl", "kəˈlɒ.nɛl", "ˈkoʊ.loʊ.nɛl"],
            "rendezvous": ["ˈrɒndeɪvuː", "rɛn.deɪ.vʊs", "rɑːn.deɪ.vʊ"],
            "debt": ["dɛt", "dɛbt", "dɪɛt"],
            "buffet": ["ˈbʊ.feɪ", "ˈbʌ.fɪt", "buː.fɛt"],
            "valet": ["ˈvæ.lɪt", "væˈleɪ", "vɑː.leɪ"],
            "yacht": ["jɒt", "jætʃ", "jɔːt"],
            "façade": ["fəˈsɑːd", "fæs.keɪd", "feɪs.æd"],
            "silhouette": ["sɪl.uˈɛt", "sɪ.lu.ɪt", "sɪ.loʊ.ɪt"],
            "pseudonym": ["ˈsuː.də.nɪm", "ˈpsjuː.dɒ.nɪm", "pseʊd.oʊ.nɪm"]
        }
        
        # Initialize session state for quiz
        if "current_question" not in st.session_state:
            st.session_state.current_question = random.choice(list(quiz_words.keys()))
            st.session_state.correct_answer = quiz_words[st.session_state.current_question][0]
            st.session_state.options = list(quiz_words[st.session_state.current_question])  # Ensure it's a list
            random.shuffle(st.session_state.options)
            st.session_state.quiz_submitted = False  # Track submission state
        
        st.write(f"🔹 How do you pronounce **{st.session_state.current_question}**?")
        
        user_answer = st.radio("Select the correct pronunciation:", st.session_state.options, key=f"quiz_{st.session_state.current_question}")
        
        if st.button("Submit Answer", key=f"submit_{st.session_state.current_question}"):
            st.session_state.quiz_submitted = True  # Mark as submitted
            if user_answer == st.session_state.correct_answer:
                st.success("🎉 Correct!")
            else:
                st.error(f"❌ Incorrect. The correct pronunciation is {st.session_state.correct_answer}")
        
        if st.session_state.quiz_submitted and st.button("Next Question", key="next_question"):
            st.session_state.current_question = random.choice(list(quiz_words.keys()))
            st.session_state.correct_answer = quiz_words[st.session_state.current_question][0]
            st.session_state.options = list(quiz_words[st.session_state.current_question])  # Ensure it's a list
            random.shuffle(st.session_state.options)
            st.session_state.quiz_submitted = False
            # st.experimental_rerun()
            st.rerun()

    

        st.markdown("---")
#         st.subheader("Phonetics")
#         st.write("""
#                  Phonetics is a branch of linguistics that studies how humans produce and perceive sounds, or in the case of sign languages, the equivalent aspects of sign. 
# Phoneticians—linguists who specialize in studying Phonetics the physical properties of speech. When you open any English dictionary, you will find some kind of signs 
# after the word, just before the meaning of the word, those signs are called Phonetics. Phonetics will help you, how to pronounce a particular word correctly. It 
# gives the correct pronunciation of a word both in British and American English. Phonetics is based on sound.

# Learning the basics of phonetics is very simple. The first or the second page of every dictionary will have an index of phonetics. If, you know to read them. That 
# is more than enough to help pronounce any word correctly.
# Once you know to use phonetics, then you don't have to go behind anybody asking them to help you, to pronounce a particular word. You can do it yourself; 
# you can even teach others and correct them when they do not pronounce a word correctly.

# Almost all people with dyslexia, however, struggle with spelling and face serious obstacles in learning to cope with this aspect of their learning disability. 
# The definition of dyslexia notes that individuals with dyslexia have "conspicuous problems" with spelling and writing, in spite of being capable in other areas 
# and having a normal amount of classroom instruction. Many individuals with dyslexia learn to read fairly well, but difficulties with spelling (and handwriting) 
# tend to persist throughout life, requiring instruction, accommodations, task modifications, and understanding from those who teach or work with the individual.
                 
#                  """)
#         st.subheader("What Causes Spelling Mistakes:")
#         st.write("""
#                  One common but mistaken belief is that spelling problems stem from a poor visual memory for the sequences of letters in words. Recent research, however, shows 
# that a general kind of visual memory plays a relatively minor role in learning to spell. Spelling problems, like reading problems, originate with language 
# learning weaknesses. Therefore, spelling reversals of easily confused letters such as b and d, or sequences of letters, such as wnet for went are manifestations 
# of underlying language learning weaknesses rather than of a visually based problem. Most of us know individuals who have excellent visual memories for pictures, 
# color schemes, design elements, mechanical drawings, maps, and landscape features, for example, but who spell poorly. The kind of visual memory necessary for spelling 
# is closely "wired in" to the language processing networks in the brain.

# Poor spellers have trouble remembering the letters in words because they have trouble noticing, remembering, and recalling the features of language that those letters 
# represent. Most commonly, poor spellers have weaknesses in underlying language skills including the ability to analyze and remember the individual sounds (phonemes) 
# in the words, such as the sounds associated with j , ch, or v, the syllables, such as la, mem, pos and the meaningful parts (morphemes) of longer words, such as sub-, 
# -pect, or -able. These weaknesses may be detected in the use of both spoken language and written language; thus, these weaknesses may be detected when someone speaks and writes.

# Like other aspects of dyslexia and reading achievement, spelling ability is influenced by inherited traits. It is true that some of us were born to be better spellers 
# than others, but it is also true that poor spellers can be helped with good instruction and accommodations.
# Dyslexic people usually spell according to their ability to correctly pronounce words phonetically, but they may not know how to spell some words. For example, 
# in ‘phonics’, they could misspell ‘Finnish’. Dyslexics often experience: difficulty reading, such as reading without reading aloud, in teens and adults. Labor-intensive 
# reading and writing that is slow and gradual. Spelling problems. Those with dyslexia may be unable to pronounce words with complete accuracy or write in ways they are 
# comfortable in any other part of the body other than at school, yet they have “conspicuous difficulties” with both of these parts. Spelling seems to be a challenge that 
# persists as a result of dyslexia, but learning how to read with the right support can improve your performance significantly. It has yet to be determined why this is. 
# Several studies show that learning difficulties lead to a significant underestimation of phonological processing and memory.
#                  """)
        


    
# with tab3:
#     st.write("Now when you click this button you will start listening 10 words one by one please pay attention and type all those words in the field below with spaces in between. System wont repeat words.")
#     start_listening = st.button("Start My test")
#     str = st.text_input("enter the words you are listening")
#     dictate_10_words(level)
#     print(str)
#     st.write("the words are completed please click enter")
#     time.sleep(5)
#     st.write(str)
#     st.write(dictated_words)
  
# @st.cache(suppress_st_warning=True)
# def bind_socket():
#     string =  random_seq()
#     random_str = st.subheader(string)
#     time.sleep(5)
#     random_str.write("")
    
    
              
# with tab3:
#     st.header("Memory Test")
#     st.write("a sequence of 5 characters will be displayed for 5 seconds please try to remember and reproduce it later.")
#     start_memory_t = st.button("Start memory Test")
#     if start_memory_t:
#         bind_socket()
        
        
        
with tab4:   
    def talk(word: str, index: int):
        """Speak a word with numbering"""
        engine = pyttsx3.init()
        phrase = f"Word {index + 1}: {word}"
        engine.say(phrase)
        engine.runAndWait()

    @st.cache_data# Streamlit 1.18+ uses st.cache_data instead of st.cache
    def get_10_word_array(level: int):
        """Fetch 10 random words based on level"""
        if level == 1:
            voc = pd.read_csv("data/intermediate_voc.csv")
        elif level == 2:
            voc = pd.read_csv("data/elementary_voc.csv")
        else:
            return []

        arr = voc.squeeze().to_numpy()
        return random.sample(list(arr), 10)

    def dictate_10_words(level: int):
        """Dictate words with numbering, stops if the tab changes"""
        words = get_10_word_array(level)
        st.session_state.dictating = True  # Track dictation state

        for i, word in enumerate(words):
            if not st.session_state.get("dictating", False):  # Stop if dictation is interrupted
                break
            talk(word, i)
            time.sleep(5)

        return words

    def levenshtein(s1, s2):
        """Calculate Levenshtein distance for spelling accuracy"""
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    # Initialize session state
    if "dictating" not in st.session_state:
        st.session_state.dictating = False
    if "dictated_words" not in st.session_state:
        st.session_state.dictated_words = []

    level = None
    st.title("📚 Dictation Practice")

    # Select standard before starting dictation
    option = st.selectbox("Select your standard", ('2nd-4th', '5th-7th'))
    if option == '2nd-4th':
        level = 2
    elif option == '5th-7th':
        level = 1

    start_dictation = st.checkbox('Start Dictation')

        # @st.cache_data  # Caching for efficiency
        # def bind_socket(level):
        #     return dictate_10_words(level)

    # Run dictation only if checkbox is checked
        # if start_dictation:  
        #     dictated_words = bind_socket(level) 

    form = st.form(key='my_form')
    w1 = form.text_input(label='word1')
    w2 = form.text_input(label='word2')
    w3 = form.text_input(label='word3')
    w4 = form.text_input(label='word4')
    w5 = form.text_input(label='word5')
    w6 = form.text_input(label='word6')
    w7 = form.text_input(label='word7')
    w8 = form.text_input(label='word8')
    w9 = form.text_input(label='word9')
    w10 = form.text_input(label='word10')
    submit_button = form.form_submit_button(label='Submit')


    @st.cache_data  # Caching for efficiency
    def bind_socket(level):
        return dictate_10_words(level)


    if start_dictation:  
        dictated_words = bind_socket(level) 



    # @st.cache
    # def bind_socket():
    # # This function will only be run the first time it's called
    #     dictated_words = dictate_10_words(level)
    #     return dictated_words
    # dictated_words = bind_socket() 
    # pr    int(dictated_words)
        



    # @st.cache_data  # Use the correct caching decorator
    # def bind_socket(level):
    #  # This function runs only once per 'level' value
    #     dictated_words = dictate_10_words(level)
    #     return dictated_words

    # if cb:  
    #     dictated_words = bind_socket(level)  # Pass 'level' as an argument


        if submit_button:
            typed_words = []
            typed_words.append(w1)
            typed_words.append(w2)
            typed_words.append(w3)
            typed_words.append(w4)
            typed_words.append(w5)
            typed_words.append(w6)
            typed_words.append(w7)
            typed_words.append(w8)
            typed_words.append(w9)
            typed_words.append(w10)

            print(typed_words)
            print(dictated_words)

            st.write("your dictation score is (lesser the better) : " , levenshtein(" ".join(typed_words) , " ".join(dictated_words)))
            st.write("dictated words: " + " ".join(dictated_words))
            st.write("typed words: " + " ".join(typed_words))


with tab5:
    

    def mental_rotation():
        st.header("🔄 Mental Rotation Puzzles")
        st.write("Rotate letters mentally to identify the correct orientation.")

        # Define rotation mapping
        rotation_map = {
            "b": "q", "d": "p", "p": "d", "q": "b",  
            "m": "w", "w": "m", "n": "u", "u": "n"
        }

        # Initialize session state
        if "original_letter" not in st.session_state or st.session_state["new_question"]:
            st.session_state["original_letter"] = random.choice(list(rotation_map.keys()))
            st.session_state["new_question"] = False  # Reset flag after selecting a new letter

        original = st.session_state["original_letter"]
        rotated = rotation_map[original]

        st.write(f"**Original Letter:** {original}")
        guess = st.selectbox("Which letter does it look like after rotation?", list(rotation_map.values()), key="rotation_guess")

        if st.button("Check Rotation"):
            if guess == rotated:
                st.success("🎉 Correct!")
            else:
                st.error(f"❌ Incorrect! The correct rotated letter is '{rotated}'.")

        # Add a "Next Question" button to reset the letter
        if st.button("Next Question"):
            st.session_state["new_question"] = True  # Set flag for new question
            st.rerun()  # Force rerun to pick a new letter

    mental_rotation()


    # List of words and their Unsplash image queries
    # Openverse API URL
    OPENVERSE_URL = "https://api.openverse.engineering/v1/images/"

    word_image_map = {
        "Apple": "apple fruit", "Banana": "banana fruit", "Car": "car vehicle", "Dog": "dog pet",
        "Elephant": "elephant animal", "Flower": "beautiful flower", "Guitar": "guitar music",
        "House": "modern house", "Ice Cream": "ice cream dessert", "Jellyfish": "jellyfish ocean",
        "Kangaroo": "kangaroo jumping", "Lion": "lion king of jungle", "Mountain": "beautiful mountain",
        "Nest": "bird nest with eggs", "Orange": "orange fruit fresh", "Penguin": "penguin on ice",
        "Queen": "queen crown royal", "Rainbow": "rainbow in sky", "Sun": "sun in blue sky",
        "Tree": "big green tree", "Umbrella": "colorful umbrella", "Violin": "violin musical instrument",
        "Waterfall": "beautiful waterfall", "Xylophone": "xylophone toy music", "Yacht": "luxury yacht",
        "Zebra": "zebra black and white", "Train": "fast train on railway", "Boat": "sailing boat on water",
        "Eagle": "eagle flying high", "Castle": "old castle architecture", "Butterfly": "colorful butterfly on flower",
        "Dolphin": "dolphin jumping in sea", "Clock": "old vintage clock", "Fire": "campfire at night",
        "Giraffe": "giraffe eating leaves", "Laptop": "modern laptop computer", "Moon": "full moon in night sky",
        "Robot": "futuristic robot", "Tiger": "tiger in jungle", "Volcano": "volcano eruption",
        "Watch": "luxury wristwatch", "Yoga": "person doing yoga", "Zeppelin": "zeppelin airship",
    }

    # Select a random word if not already set
    if "current_word" not in st.session_state:
        st.session_state["current_word"] = random.choice(list(word_image_map.keys()))

    word = st.session_state["current_word"]
    image_query = word_image_map[word]

    # Fetch image from Openverse API
    params = {"q": image_query, "page_size": 5}
    response = requests.get(OPENVERSE_URL, params=params)

    # Extract multiple image URLs
    image_urls = []
    if response.status_code == 200:
        json_data = response.json()
        if "results" in json_data and json_data["results"]:
            image_urls = [result["url"] for result in json_data["results"]]

    # Default image if no results
    if not image_urls:
        image_urls = ["https://via.placeholder.com/400x300.png?text=No+Image+Found"]

    st.header("🖼️ Word-Picture Association")
    st.write(f"Match the word with the correct image.")

    # Display images in a grid (max 5 images)
    cols = st.columns(min(len(image_urls), 5))
    for i, url in enumerate(image_urls[:5]):  # Show only available images
        with cols[i]:
            st.image(url, use_container_width=True)

    # User selection
    user_choice = st.selectbox("Which word matches the image?", list(word_image_map.keys()))

    # Button for checking answer
    if st.button("Check Answer", key="check_answer_button"):
        if user_choice == word:
            st.success("🎉 Correct!")
        else:
            st.error(f"❌ Incorrect! The correct match is **{word}**.")

    # Button to load next question
    if st.button("Next Question", key="next_question_button"):
        st.session_state["current_word"] = random.choice(list(word_image_map.keys()))
        st.rerun()  # Immediately refresh the UI



    SENTENCES = [
        "Practice makes perfect",
        "Reading is a great habit",
        "Never stop learning",
        "Knowledge is power",
        "Stay curious and keep exploring",
        "Hard work pays off",
        "Success is a journey, not a destination",
        "Believe in yourself and never give up",
        "Great things take time",
        "Every day is a new opportunity"
    ]

    def reverse_reading():
        st.header("🔄 Reverse Reading Practice")

        # Initialize session state for sentence tracking
        if "current_sentence" not in st.session_state:
            st.session_state.current_sentence = random.choice(SENTENCES)

        sentence = st.session_state.current_sentence
        reversed_sentence = sentence[::-1]  # Reverse character-by-character

        st.write("Original Sentence:", sentence)

        # User input for reversed sentence
        user_input = st.text_input("Type the sentence in reverse:")

        # Check button
        if st.button("Check Reverse Reading"):
            if user_input.strip() == reversed_sentence:
                st.success("🎉 Correct!")
            else:
                st.error(f"❌ Try Again! The correct reverse is: {reversed_sentence}")

        # Button to generate a new random sentence
        if st.button("Next Sentence"):
            st.session_state.current_sentence = random.choice(SENTENCES)
            st.rerun()  # Refresh the app to update the displayed sentence

    reverse_reading()

    # Word Tracing Practice
    from PIL import Image, ImageDraw, ImageFont

    # Function to generate reference letter image
    # Configure Tesseract path (Update this if necessary)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path if needed

    st.header("📝 Word Tracing Practice")

    # Select letter to trace (A-Z, a-z)
    letters = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]  # A-Z & a-z
    letter = st.selectbox("Choose a letter to trace:", letters)

    st.write("🖌️ Trace the letter below:")

    # Create a drawable canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=5,
        stroke_color="black",
        background_color="white",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    # Button to check drawing
    if st.button("Check My Drawing"):
        if canvas_result.image_data is not None:
            # Convert drawn image to grayscale
            img = np.array(canvas_result.image_data, dtype=np.uint8)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

            # Apply thresholding to improve OCR accuracy
            _, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(img_thresh, config="--psm 6").strip()

            st.write(f"📝 Extracted Text: **{extracted_text}**")

            # Check if the extracted text contains the selected letter
            if letter in extracted_text:
                st.success("🎉 Correct! Your tracing matches the given letter.")
            else:
                st.error("❌ Incorrect! Your tracing does not match the given letter.")

            # Debugging: Show processed images
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_gray, caption="Your Drawing", use_container_width=True)
            with col2:
                st.image(img_thresh, caption="Processed for OCR", use_container_width=True)

with tab6:
    st.header("About APP")
    st.write("""
    Dyslexia, also known as reading disorder, is a disorder characterized by reading below the expected level for ones age. 
    Different people are affected to different degrees.
    The common symptoms include: Frequently making the same kinds of mistakes, like reversing letters, Having poor spelling, like spelling the same word correctly and 
    incorrectly in the same exercise, Having trouble remembering how words are spelled and applying spelling rules in writing, etc.

    Based on the spelling, grammatic, contextual and phonetics error the app predicts whether the person with the wrting has 
    dyslexia or not. 
    """)
    st.subheader("Average corrections is less for a non-dyslexic child when compared to dyslexic child")
    st.image("images\percentage_of_corrections.jpg")
    
    st.subheader("Spelling accuracy for a dyslexic and a non-dyslexic child")
    st.image("images\spelling_accuracy.jpg")
    
    st.subheader("Average Phonetic accuracy comparision between a dyslexic and a non-dyslexic child ")
    st.image("images\percentage_of_phonetic_accuraccy.jpg")

