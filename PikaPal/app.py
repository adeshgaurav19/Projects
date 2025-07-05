import streamlit as st
import requests
import random

# Page configuration
st.set_page_config(
    page_title="PikaPal - Fun & Magical Adventures",
    page_icon="🌟",
    layout="wide"
)

# Enhanced CSS with better readability, animations, and kid-friendly design
page_style = """
<style>
    .stApp {
        background: 
          linear-gradient(to bottom, #fff3e0, #ffd1dc) no-repeat fixed, 
          url("https://cdn.pixabay.com/photo/2017/08/30/07/56/background-2696336_1280.jpg") no-repeat center;
        background-size: cover;
        color: #333333;
    }
    h1, h2, h3 {
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #ff4081;
        text-shadow: 3px 3px #fff176;
        margin-bottom: 25px;
    }
    .story-box {
        padding: 30px;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 25px;
        border: 5px solid #ffeb3b;
        color: #d81b60;
        font-size: 26px;
        font-family: "Comic Sans MS", cursive, sans-serif;
        margin-bottom: 30px;
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
        line-height: 1.6;
    }
    .sparkles {
        font-size: 70px;
        animation: sparkle 1.8s infinite alternate;
        color: #ffeb3b;
        display: inline-block;
    }
    @keyframes sparkle {
        0%   { transform: scale(1.0) rotate(0deg); color: #ffeb3b; }
        50%  { transform: scale(1.3) rotate(10deg); color: #ff9100; }
        100% { transform: scale(1.5) rotate(20deg); color: #ff4081; }
    }
    .big-button {
        font-size: 28px !important;
        padding: 20px 40px !important;
        border-radius: 20px !important;
        background-color: #ffeb3b !important;
        color: #d81b60 !important;
        font-family: 'Comic Sans MS', cursive, sans-serif !important;
        border: 4px solid #ff4081 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        display: block !important;
        margin: 10px auto !important;
        width: fit-content !important;
    }
    .big-button:hover {
        background-color: #ff4081 !important;
        color: #fff !important;
        transform: scale(1.1) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.25) !important;
    }
    .game-container {
        text-align: center;
        background-color: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        border: 5px solid #ff4081;
        color: #d81b60;
        font-family: "Comic Sans MS", cursive, sans-serif;
        margin: 30px 0;
        font-size: 22px;
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    .game-title {
        color: #9c27b0;
        font-size: 32px;
        margin-bottom: 20px;
        text-shadow: 2px 2px #ffeb3b;
    }
    .game-button {
        background-color: #66bb6a !important;
        color: white !important;
        font-family: 'Comic Sans MS', cursive, sans-serif !important;
        font-size: 20px !important;
        padding: 12px 25px !important;
        border-radius: 15px !important;
        border: 3px solid #388e3c !important;
        transition: all 0.3s !important;
    }
    .game-button:hover {
        background-color: #43a047 !important;
        transform: scale(1.1) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    .stTextInput > div > div > input {
        font-size: 22px !important;
        border: 3px solid #ff9800 !important;
        border-radius: 15px !important;
        padding: 12px !important;
    }
    .stTextInput > div > div > input:focus {
        border: 4px solid #ff4081 !important;
        box-shadow: 0 0 10px #ff9800 !important;
    }
    .stSelectbox > div > div > div {
        font-size: 22px !important;
        border: 3px solid #ff9800 !important;
        border-radius: 15px !important;
    }
    .footer {
        text-align: center;
        font-size: 20px;
        margin-top: 40px;
        color: #d81b60;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 20px;
    }
    .clue-text {
        font-size: 26px;
        color: #9c27b0;
        background-color: rgba(255, 240, 150, 0.6);
        padding: 15px;
        border-radius: 15px;
        margin: 20px 0;
        border: 3px dashed #ff9800;
    }
    .success-message {
        font-size: 24px !important;
        padding: 20px !important;
        border-radius: 15px !important;
        background-color: rgba(139, 195, 74, 0.3) !important;
        color: #388e3c !important;
        border: 3px solid #8bc34a !important;
    }
    .error-message {
        font-size: 24px !important;
        padding: 20px !important;
        border-radius: 15px !important;
    }
    .toggle-container {
        display: flex;
        justify-content: center;
        margin: 25px 0;
        gap: 20px;
    }
    .bounce {
        animation: bounce 2s infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    .sing-along {
        font-size: 26px;
        line-height: 1.8;
        background: linear-gradient(to right, #ff4081, #9c27b0);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: bold;
        padding: 20px;
        border: 4px dashed #ff9800;
        border-radius: 20px;
        background-color: rgba(255, 255, 255, 0.85);
    }
    .color-img {
        border-radius: 20px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.25);
        margin: 20px auto;
        border: 4px solid #ff9800;
        transition: all 0.3s;
    }
    .color-img:hover {
        transform: scale(1.03);
    }
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Session state initialization
if 'story_count' not in st.session_state:
    st.session_state['story_count'] = 0
if 'current_game' not in st.session_state:
    st.session_state['current_game'] = None
if 'quiz_animal' not in st.session_state:
    st.session_state['quiz_animal'] = None
if 'quiz_clue_index' not in st.session_state:
    st.session_state['quiz_clue_index'] = 0
if 'game_success' not in st.session_state:
    st.session_state['game_success'] = False

# Header
st.markdown("<h1>🌈 Welcome to PikaPal! 🌈</h1>", unsafe_allow_html=True)
st.markdown("<h3>Your Magical Friend for Stories, Games, & Fun! 🎉</h3>", unsafe_allow_html=True)

# Centered Pikachu with bounce animation
pikachu_urls = [
    "https://huggingface.co/spaces/adzee19/StoryTeller_Agent/resolve/main/pika.gif"
]
st.markdown("<div style='text-align: center;' class='bounce'>", unsafe_allow_html=True)
try:
    st.image(
        random.choice(pikachu_urls),
        caption="Hi! I’m PikaPal, bouncing with joy for you!",
        width=280
    )
except:
    st.markdown("<h3>Oops! Pikachu is bouncing somewhere else today! Let’s play anyway! 😺</h3>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Story Section
st.markdown("## ✨ Story Adventure Time ✨")
theme = st.text_input(
    "What kind of story would you like?",
    placeholder="Cats, Dragons, Stars, Pirates, Unicorns...",
    key="story_input"
)

st.markdown("### Your Story Stars 🌠")
progress_percent = min(st.session_state['story_count'] * 10, 100)
star_bar = st.progress(progress_percent)
st.markdown(f"<p style='text-align: center; font-size: 20px;'>You’ve collected {st.session_state['story_count']} sparkly stars!</p>", unsafe_allow_html=True)

N8N_WEBHOOK_URL = "https://adzee.app.n8n.cloud/webhook/1597d563-462c-42a7-87d3-9c6b416bea3d"

if st.button("✨ Tell Me a Story! ✨", key="story_button", help="Click for a magical story!", type="primary"):
    st.markdown("""
        <script>
            const buttons = document.querySelectorAll('button[kind="primary"]');
            buttons.forEach(button => {
                button.classList.add('big-button');
            });
        </script>
    """, unsafe_allow_html=True)
    with st.spinner("Boing! Creating a magical story just for you... 🧙‍♂️"):
        try:
            story_theme = theme if theme.strip() else "magical adventure"
            response = requests.post(N8N_WEBHOOK_URL, json={"theme": story_theme}, timeout=15)
            if response.status_code == 200:
                story = response.json().get("story", "Oops! The story fairy is napping. Try again!")
                st.markdown(f"""
                <div class="story-box">
                    {story}
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                st.session_state['story_count'] += 1
                star_bar.progress(min(st.session_state['story_count'] * 10, 100))
                fun_messages = [
                    "Wowee! That was pawsitively purrfect! 🐾",
                    "You’re a story superhero! 🦸‍♂️",
                    "Zoom! What a blast-off adventure! 🚀",
                    "Magic hugs for you! 🤗",
                    "Yummy! Sweeter than rainbow candy! 🍭",
                    "Twinkle! You’re a star explorer! ⭐",
                ]
                st.success(random.choice(fun_messages), icon="🎉")
            else:
                st.error("Oopsie! The story machine is sleepy. Try again soon!", icon="😴")
        except requests.exceptions.RequestException:
            st.error("Whoops! The story magic got lost. Let’s play a game instead!", icon="🎮")

st.markdown("---")

# Game Functions
def reset_games():
    st.session_state['quiz_animal'] = None
    st.session_state['quiz_clue_index'] = 0
    st.session_state['game_success'] = False

def switch_game(game_name):
    reset_games()
    st.session_state['current_game'] = game_name

# Game Section
st.markdown("## 🎮 Super Fun Games Zone 🎮")
st.markdown("<p style='text-align: center; font-size: 22px;'>Pick a game to play, little adventurer!</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🕹️ Animal Guess", key="guess_game_btn", help="Guess the mystery animal!", use_container_width=True):
        switch_game("guess")
with col2:
    if st.button("🎨 Color Magic", key="color_game_btn", help="Paint with your imagination!", use_container_width=True):
        switch_game("color")
with col3:
    if st.button("🎵 Sing-Along", key="sing_along_btn", help="Sing a happy song!", use_container_width=True):
        switch_game("sing")

# Game Container
if st.session_state['current_game']:
    st.markdown("<div class='game-container'>", unsafe_allow_html=True)

    # Animal Guessing Game
    if st.session_state['current_game'] == "guess":
        st.markdown("<h2 class='game-title'>🕹️ Guess the Animal!</h2>", unsafe_allow_html=True)
        st.write("I’m a mystery animal! Can you guess me with these clues?")
        
        if st.session_state['quiz_animal'] is None:
            animals = [
                {"name": "Elephant", "clues": ["I have a long trunk!", "I’m super big!", "I love splashing water!"]},
                {"name": "Giraffe", "clues": ["I have a tall neck!", "I munch tree leaves!", "I’m spotty!"]},
                {"name": "Kangaroo", "clues": ["I hop hop hop!", "I have a pouch!", "I’m from Australia!"]},
                {"name": "Penguin", "clues": ["I waddle on ice!", "I swim, not fly!", "I love the cold!"]},
                {"name": "Zebra", "clues": ["I have black and white stripes!", "I run fast!", "I live in Africa!"]},
                {"name": "Tiger", "clues": ["I have orange fur!", "I have black stripes!", "I’m a big cat!"]},
                {"name": "Monkey", "clues": ["I love bananas!", "I can climb trees!", "I’m very playful!"]},
                {"name": "Owl", "clues": ["I’m awake at night!", "I have big eyes!", "I say 'hoot hoot'!"]},
            ]
            st.session_state['quiz_animal'] = random.choice(animals)
        
        animal_data = st.session_state['quiz_animal']
        clue_index = st.session_state['quiz_clue_index']
        
        st.markdown(f"<div class='clue-text'>Clue #{clue_index + 1}: {animal_data['clues'][clue_index]}</div>", unsafe_allow_html=True)
        
        if clue_index < len(animal_data['clues']) - 1 and not st.session_state['game_success']:
            if st.button("🔎 Next Clue, Please!", key="next_clue_btn", type="secondary"):
                st.session_state['quiz_clue_index'] += 1
        
        if not st.session_state['game_success']:
            guess = st.text_input("Who am I? (Type your guess!)", key="guess_input", placeholder="What’s your guess?")
            if st.button("🎯 Guess It!", key="guess_button", type="primary"):
                if guess.strip().lower() == animal_data['name'].lower():
                    st.session_state['game_success'] = True
                    st.success(f"🎉 Yay! You guessed it! I’m a {animal_data['name']}! Woohoo!", icon="🎊")
                    st.balloons()
                else:
                    st.error("Oopsie-daisy! Not quite right. Guess again!", icon="🧐")
        
        if st.session_state['game_success']:
            if st.button("🎮 Play Again!", key="play_again_btn"):
                reset_games()
                st.session_state['quiz_animal'] = None

    # Color Magic Game
    elif st.session_state['current_game'] == "color":
        st.markdown("<h2 class='game-title'>🎨 Color Magic!</h2>", unsafe_allow_html=True)
        st.write("Pick a color and watch the magic happen!")
        
        colors = {
            "Red": "#ff4444", "Blue": "#4444ff", "Yellow": "#ffff44", "Green": "#44ff44",
            "Purple": "#aa44ff", "Pink": "#ff66cc", "Orange": "#ffaa44", "Rainbow": "linear-gradient(to right, #ff4444, #4444ff, #ffff44)",
            "Gold": "#ffd700", "Silver": "#c0c0c0", "Magic Sparkles": "linear-gradient(to right, #ffeb3b, #ff4081)"
        }
        chosen_color = st.selectbox("Pick a magical color!", list(colors.keys()), key="color_select")
        
        images = [
            {"url": "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832_1280.jpg", "name": "Forest"},
            {"url": "https://cdn.pixabay.com/photo/2016/04/18/22/05/seashells-1337565_1280.jpg", "name": "Seashells"},
            {"url": "https://cdn.pixabay.com/photo/2017/02/01/10/41/feathers-2029112_1280.png", "name": "Feathers"},
            {"url": "https://cdn.pixabay.com/photo/2017/07/24/19/57/tiger-2535888_1280.png", "name": "Tiger"},
        ]
        selected_image = random.choice(images)
        
        st.markdown(f"<h3>Imagine this {selected_image['name']} in {chosen_color}!</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='position: relative; display: inline-block;'>
                <img src="{selected_image['url']}" class="color-img" width="400">
                <div style='position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
                            background: {colors[chosen_color]}; opacity: 0.3; border-radius: 20px;'></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.write(f"✨ Wow! It’s {chosen_color}! What would you add to make it even cooler? ✨")
        drawing_idea = st.text_input("My magical idea:", key="drawing_idea", placeholder="Stars? A unicorn? A rocket?")
        if drawing_idea:
            st.write(f"Boing! Your {chosen_color} {drawing_idea} makes it SO awesome!")
            st.balloons()

    # Sing-Along Game
    elif st.session_state['current_game'] == "sing":
        st.markdown("<h2 class='game-title'>🎵 Sing-Along Time!</h2>", unsafe_allow_html=True)
        st.write("Let’s sing a silly song together! Tap to see the words!")
        
        songs = [
            {
                "title": "Twinkle PikaPal",
                "lyrics": ["Twinkle, twinkle, Pika-star! 🌟", "Dancing high is where you are! 💃", 
                           "Up above with fluffy cheer! ☁️", "PikaPal is always near! 👋"]
            },
            {
                "title": "PikaBus Wheels",
                "lyrics": ["The wheels on the PikaBus go round and round! 🚌", "Round and round, all day long! 🔄", 
                           "Zooming through the magic town! 🏙️", "Sing with me, it’s so much fun! 🎶"]
            },
            {
                "title": "Happy Claps",
                "lyrics": ["If you’re happy and you know it, clap your hands! 👏", "Clap clap clap, hooray! 😃", 
                           "If you’re happy and you know it, stomp your feet! 👣", "Stomp stomp, yay yay yay! 🎉"]
            }
        ]
        
        if 'current_song' not in st.session_state:
            st.session_state['current_song'] = random.choice(songs)
            st.session_state['revealed_lines'] = [False] * len(st.session_state['current_song']['lyrics'])
        
        current_song = st.session_state['current_song']
        st.markdown(f"<h3>🎤 {current_song['title']} 🎤</h3>", unsafe_allow_html=True)
        
        for i, line in enumerate(current_song['lyrics']):
            if st.session_state['revealed_lines'][i]:
                st.markdown(f"<div class='sing-along'>{line}</div>", unsafe_allow_html=True)
            elif st.button(f"🎵 Sing Line {i+1}!", key=f"sing_line_{i}"):
                st.session_state['revealed_lines'][i] = True
        
        all_revealed = all(st.session_state['revealed_lines'])
        if all_revealed:
            st.success("🎵 Woohoo! You sang it all! You’re a singing superstar! 🌟", icon="🎤")
            if st.button("🎶 New Song, Please!", key="new_song_btn"):
                del st.session_state['current_song']
                del st.session_state['revealed_lines']

    st.markdown("</div>", unsafe_allow_html=True)

# Reset Game Button
if st.session_state['current_game']:
    if st.button("🏠 Back to Games Menu", key="reset_games_btn"):
        st.session_state['current_game'] = None
        reset_games()

# Footer
st.markdown("<div style='text-align: center;'><span class='sparkles'>✨</span> <span class='sparkles'>🌟</span> <span class='sparkles'>✨</span></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>🌈 Made with magic and giggles for little adventurers! 🌈<br>© PikaPal 2025</div>", unsafe_allow_html=True)