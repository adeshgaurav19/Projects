import streamlit as st
from story_generator import generate_story
from story_evaluator import is_story_safe

# Streamlit UI
st.title("AI Storyteller for Kids 📖✨")

theme = st.text_input("Enter a theme (e.g., space, animals, pirates)")
age = st.slider("Select Age", min_value=3, max_value=10, value=4)

if st.button("Generate Story"):
    if theme:
        story = generate_story(theme, age)
        evaluation = is_story_safe(story)

        st.subheader("Generated Story:")
        if evaluation["safe"]:
            st.write(story)
            st.success("✅ The story is safe for kids!")
        else:
            st.warning(f"⚠️ The story might not be suitable: {evaluation['reason']}")
            st.subheader("Censored Story (Filtered for Kids):")
            st.write(evaluation["filtered_story"])
    else:
        st.warning("Please enter a theme!")
