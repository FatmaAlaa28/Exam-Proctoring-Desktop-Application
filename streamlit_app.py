import streamlit as st
import subprocess
import pandas as pd

import numpy as np
import time
from streamlit_extras.badges import badge
from streamlit_extras.app_logo import add_logo

# Set page configuration (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="Cheat System", page_icon="")

# Custom CSS to style the buttons
# Custom CSS to style the sidebar image and center it
st.markdown("""
<style>
/* Centering the logo in the sidebar */
.sidebar .sidebar-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.sidebar .sidebar-content img {
    width: 100px;  /* Adjust the size of the logo */
    margin-bottom: 20px; /* Optional margin for spacing */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Custom CSS to make buttons larger and change their color */
.stButton > button {
    height: 50px; /* Adjust height */
    width: 70%; /* Full width of the column */
    font-size: 50px; /* Larger font size */
    background-color: #A97A5B; /* Beige background color */
    color: black; /* Text color */
    border-radius: 10px; /* Rounded corners */
    border: 2px solid #A97A5B; /* Border */
    margin: 10px 0; /* Add some margin */
}
.stButton > button:hover {
    background-color: #BF9A7A; /* Lighter beige on hover */
    border: 2px solid #A97A5B; /* Keep border on hover */
     color: black;
}
</style>
""", unsafe_allow_html=True)

# def example():
#     if st.checkbox("Use url", value=True):
#         add_logo("images/Screenshot_2025-01-27_154902-removebg-preview.png")
#     else:
#         add_logo("images/Screenshot_2025-01-27_154902-removebg-preview.png", height=300)
#     st.write("👈 Check out the cat in the nav-bar!")

# Sidebar navigation
# Sidebar navigation
st.sidebar.image(r"images\WhatsApp_Image_2025-02-07_at_5.01.26_PM-removebg-preview.png", width=100)
if st.sidebar.button("Home"):
    st.session_state.page = "home"
if st.sidebar.button("Contact Us"):
    st.session_state.page = "contact"
if st.sidebar.button("Services"):
    st.session_state.page = "services"
if st.sidebar.button("About"):
    st.session_state.page = "about"
if st.sidebar.button("Master"):  # Add a button for the master page
    st.session_state.page = "master"
if st.sidebar.button("Student"):  # Add a button for the master page
    st.session_state.page = "student"


# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Main content based on navigation
if st.session_state.page == "home":
    st.markdown("<h1 style='color:#563B30;'><span style='color:#563B30;'>Online Exam Proctoring System</span> 💻</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border:2px solid #A97A5B;'>", unsafe_allow_html=True)

    left, middle, right = st.columns(3)

    # Button for Student
    if left.button("🎓 I am Student", use_container_width=True):
        st.session_state.page = "student"

    # Button for Teacher
    if middle.button("👩‍🏫 I am Master", use_container_width=True):
        st.session_state.page = "Master"

elif st.session_state.page == "student":
         st.markdown("<h1 style='text-align: center; margin-bottom:50px;'>🎓 Student </h1>", unsafe_allow_html=True)

# Welcome Statement
         st.markdown("<h3 ; margin-bottom:80px;margin-bottom:50px;'>Welcome to the student section!</h3>", unsafe_allow_html=True)

# Start Exam Button with Exam Icon
         if st.button("📝 Start Exam"):
              with st.spinner("Starting exam..."):
                 result = subprocess.run(["python", "main4.py"], capture_output=True, text=True)
                 st.write(result.stdout)  # Display the output from the script if needed


elif st.session_state.page == "master":
    st.markdown("<h1 style='text-align: center;margin-bottom:50px;'>👨‍🏫 Master </h1>", unsafe_allow_html=True)
    st.markdown("<h3 margin-bottom:50px;'>Welcome to the teacher section!</h3>", unsafe_allow_html=True)

    # Start Exam Button with a Cheating Icon
    if st.button("🚨 Start Mastering"):
        with st.spinner("Start Mastering..."):
            # Run the script (eye.py) and capture the output
            result = subprocess.run(["python", "master.py"], capture_output=True, text=True)
            st.write(result.stdout)  # Display the output from the script if needed




elif st.session_state.page == "contact":
    st.markdown("<h1 style='text-align: center;'>📞 Contact Us</h1>", unsafe_allow_html=True)

    # Fake contact details with icons
    st.markdown("<h3 style='text-align: center;'>Reach out to us!</h3>", unsafe_allow_html=True)

    contact_column1, contact_column2 = st.columns(2)

    with contact_column1:
        # Phone Icon and fake number
        st.markdown("📱 **Phone**: +1 234 567 890")

        # Email Icon and fake email
        st.markdown("✉️ **Email**: contact@onlineexamproctor.com")

    with contact_column2:
        # Instagram Icon and fake Instagram link
        st.markdown("📸 **Instagram**: [@OnlineExamProctor](https://www.instagram.com/onlineexamproctor)")

        # Twitter Icon and fake Twitter link
        st.markdown("🐦 **Twitter**: [@OnlineExamProctor](https://twitter.com/onlineexamproctor)")

    st.markdown("<hr style='border:2px solid #A97A5B;'>", unsafe_allow_html=True)


elif st.session_state.page == "services":
    st.markdown("<h1 style='text-align: center;'>💼 Our Services</h1>", unsafe_allow_html=True)

    # Introduction to the Cheating Detection System
    st.markdown("<h3 style='text-align: center;'>How Our Cheating Detection System Works</h3>", unsafe_allow_html=True)

    st.write("""
    Our **Online Exam Proctoring** system provides an innovative solution to ensure fairness in online exams.
    By using advanced eye-tracking and behavior analysis, we can detect any signs of cheating, such as:
    - **Looking Away** from the screen (indicating potential cheating).
    - **Facial Expressions** that suggest dishonesty.
    - **Unexpected Movements** (such as talking to someone off-camera).

    We continuously improve our system to provide accurate and reliable cheating detection for educators.
    """)

    # Our Services
    st.markdown("<h3 style='text-align: center;'>Our Offered Services</h3>", unsafe_allow_html=True)

    # Service Details
    services_column1, services_column2 = st.columns(2)

    with services_column1:
        st.markdown("🔍 **Cheating Detection**: We use real-time analysis to identify cheating during exams.")
        st.markdown("🛡️ **Exam Proctoring**: Our system ensures exam integrity by monitoring students.")

    with services_column2:
        st.markdown("📊 **Detailed Reports**: After each exam, we provide comprehensive reports on detected incidents.")
        st.markdown("📅 **Customizable Settings**: Educators can configure exam monitoring parameters to meet their needs.")

    st.markdown("<hr style='border:2px solid #A97A5B;'>", unsafe_allow_html=True)


elif st.session_state.page == "about":
    # Page Title with Icon
    st.markdown("<h1 style='text-align: center; color: #A97A5B;'>🌟 About Us</h1>", unsafe_allow_html=True)

    # Subheading
    st.markdown("<h3 style='text-align: center;'>Our Mission and Vision</h3>", unsafe_allow_html=True)

    # About Us Description
    st.markdown("""
    We are a team of passionate developers and educators committed to making online exams more secure and reliable. Our **Online Exam Proctoring System** is designed to maintain the integrity of online education, ensuring a fair and honest exam experience for all students.

    ### Our Mission:
    - **Promote Integrity**: Our primary goal is to promote integrity and fairness in the education system by detecting cheating during online exams.
    - **Innovative Solutions**: We use cutting-edge technology like **eye-tracking** and **behavioral analysis** to detect any signs of cheating.

    ### Our Vision:
    - **Reliable Proctoring**: To build a reliable, user-friendly system that helps both educators and students maintain the highest standards of integrity during exams.
    - **Global Impact**: We aim to create a global solution that supports educational institutions around the world.

    We believe that education should be fair and accessible to everyone, and we strive to make that a reality through our innovative system.
    """)

    # Image Section (Optional)
    st.markdown("<hr style='border:2px solid #A97A5B;'>", unsafe_allow_html=True)

    # Custom CSS for background and text styling
    st.markdown("""
    <style>
    .about-section {
        background-color: #F9F9F9; /* Light grey background */
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .about-heading {
        color: #A97A5B;
        font-weight: bold;
        text-align: center;
    }
    .about-text {
        font-size: 18px;
        color: #333;
        text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="about-heading">Why Choose Us?</h2>', unsafe_allow_html=True)
    st.markdown("""
    We offer a unique approach to online exam proctoring that is both **secure** and **non-invasive**.
    Our system ensures that the student’s privacy is maintained while providing accurate detection of potential cheating activities.

    ### Key Benefits:
    - **Reliable detection**: Using advanced machine learning techniques for accurate and real-time detection.
    - **User-friendly interface**: Simple and easy for both students and educators.
    - **Customizable settings**: Allows educators to tailor the system to their needs.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border:2px solid #A97A5B;'>", unsafe_allow_html=True)
