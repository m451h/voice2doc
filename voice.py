import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
from openai import OpenAI
import os
from datetime import datetime
import json
import base64
from dotenv import load_dotenv

# 🔑 API Configuration - Load from .env file
load_dotenv()

# Get API key and base URL from environment variables
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key or not base_url:
    st.error("❌ خطا: لطفاً فایل .env را ایجاد کنید و OPENAI_API_KEY و OPENAI_BASE_URL را تنظیم کنید.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = base_url

client = OpenAI()

# ============================================
# ENHANCED SYSTEM PROMPTS
# ============================================

PATIENT_ANALYSIS_PROMPT = """
شما یک سیستم هوش مصنوعی پزشکی پیشرفته هستید که به عنوان دستیار اورژانس عمل می‌کنید.

## اطلاعات بیمار:
علائم گزارش شده: {symptoms}
تاریخ و زمان: {timestamp}

## وظایف شما (به ترتیب اولویت):

### 1️⃣ ارزیابی فوریت (TRIAGE)
بر اساس پروتکل‌های استاندارد اورژانس، وضعیت را طبقه‌بندی کنید:

🔴 **فوریت بحرانی (قرمز)**: نیاز به مراجعه فوری به اورژانس (0-1 ساعت)
🟡 **فوریت متوسط (زرد)**: مراجعه به پزشک در 24 ساعت آینده
🟢 **غیرفوری (سبز)**: قابل پیگیری با پزشک خانواده

### 2️⃣ تشخیص‌های احتمالی (Differential Diagnosis)
لیست 3-5 تشخیص محتمل با درصد احتمال به این صورت:

**تشخیص اول (احتمال XX%):**
- نام بیماری: [نام به فارسی و انگلیسی]
- دلیل: [چرا این تشخیص محتمل است]
- علائم کلیدی: [علائمی که با این بیماری همخوانی دارند]

### 3️⃣ علائم خطر (Red Flags) ⚠️
اگر هر یک از این علائم ظاهر شد، فوراً به اورژانس مراجعه کنید:
- [لیست علائم خطرناک مرتبط]

### 4️⃣ سوالات تکمیلی برای تشخیص دقیق‌تر
برای تشخیص بهتر، لطفاً به این سوالات پاسخ دهید:
1. [سوال مهم اول]
2. [سوال مهم دوم]
3. [سوال مهم سوم]

### 5️⃣ توصیه‌های اولیه
📌 اقدامات خانگی:
- [توصیه 1]
- [توصیه 2]

💊 داروهای بدون نسخه (در صورت نیاز):
- [دارو با دوز و هشدارها]

🚫 موارد ممنوع:
- [کارهایی که نباید انجام دهد]

### 6️⃣ برنامه پیگیری
- بررسی مجدد علائم در [مدت زمان]
- در صورت بدتر شدن: [راهنمایی]

---
⚠️ **مهم:** این ارزیابی جایگزین معاینه پزشکی نیست.
📊 **سطح اطمینان تحلیل:** [پایین/متوسط/بالا]
"""

DOCTOR_QUESTIONS_PROMPT = """
شما یک پزشک متخصص با تجربه بالا هستید.

## اطلاعات موجود:
علائم بیمار: {symptoms}

## وظیفه:
برای تشخیص دقیق، سوالات تکمیلی حرفه‌ای بپرسید.

### قالب سوالات:

**📋 بخش 1: تاریخچه دقیق علائم**
1. **زمان شروع:** این علائم از چه زمانی شروع شده؟ آیا ناگهانی بوده؟
2. **الگوی علائم:** دائمی است یا موقت؟ چه زمانی بدتر می‌شود؟

**📋 بخش 2: شدت و کیفیت**
3. **مقیاس شدت:** در مقیاس 1 تا 10 چقدر است؟
4. **نوع احساس:** تیز، سوزاننده، فشاری، یا کند کننده؟

**📋 بخش 3: عوامل تشدیدکننده**
5. **چه چیزی وضعیت را بهتر/بدتر می‌کند؟**

**📋 بخش 4: علائم همراه**
6. **سایر علائم:** تب، لرز، تغییر اشتها، وزن، خواب؟

**📋 بخش 5: سابقه پزشکی**
7. **تاریخچه:** آیا قبلاً علائم مشابه داشته‌اید؟ بیماری زمینه‌ای؟ داروهای مصرفی؟

**📋 بخش 6: سبک زندگی**
8. **محیط:** سفر اخیر؟ تماس با بیماران؟ تغییر در رژیم غذایی؟

---
💡 **هدف:** با پاسخ به این سوالات، تشخیص دقیق‌تری ممکن می‌شود.
"""

EMERGENCY_PROTOCOL = """
🚨 پروتکل اورژانس - ارزیابی سریع

علائم: {symptoms}

## ❗ بررسی فوری علائم خطرناک:

### ⚠️ Red Flags (علائم خطر فوری):

**قلبی-عروقی:**
✋ درد قفسه سینه + تعریق → حمله قلبی احتمالی
✋ تپش قلب شدید + بیهوشی

**عصبی:**
✋ فلج ناگهانی یک طرفه → سکته مغزی
✋ سردرد رعدآسا شدید
✋ اختلال هوشیاری

**تنفسی:**
✋ تنگی نفس شدید
✋ کبودی لب‌ها
✋ سرفه خونی

**گوارشی:**
✋ درد شکم ناگهانی و شدید
✋ استفراغ خونی

**سایر:**
✋ خونریزی شدید
✋ واکنش آلرژیک شدید
✋ تب بالای 40 درجه

## تصمیم‌گیری:

**وجود علائم بالا:**
🚨 تماس فوری با اورژانس 115

**وضعیت پایدار:**
📞 مشاوره پزشکی
"""


# ============================================
# HELPER FUNCTIONS
# ============================================

def ask_model(prompt, model="gpt-4o-mini"):
    """Send request to AI model with enhanced error handling"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ خطا در ارتباط با مدل: {str(e)}"


def save_consultation(symptoms, analysis, role):
    """Save consultation to session history"""
    if "consultation_history" not in st.session_state:
        st.session_state.consultation_history = []
    
    consultation = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "symptoms": symptoms,
        "analysis": analysis
    }
    st.session_state.consultation_history.append(consultation)


def format_prompt(template, **kwargs):
    """Format prompt template with parameters"""
    return template.format(**kwargs)


# ============================================
# STREAMLIT UI CONFIGURATION
# ============================================

st.set_page_config(
    page_title="دستیار تشخیص بیماری هوشمند داویس",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
    }
    .header-logo {
        height: 80px;
        width: auto;
        border-radius: 10px;
    }
    .header-title {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-weight: bold;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================

# Logo configuration - update this path to your logo file
LOGO_PATH = "logo.png"  # Change this to your logo file path

# Display header with title
st.markdown(
    f'<div class="main-header">'
    f'<h1 style="margin: 0; display: flex; align-items: center;">دستیار تشخیص بیماری هوشمند داویس</h1>'
    f'</div>',
    unsafe_allow_html=True
)

# Initialize session state
if "patient_symptoms" not in st.session_state:
    st.session_state.patient_symptoms = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "consultation_history" not in st.session_state:
    st.session_state.consultation_history = []

# Sidebar
with st.sidebar:
    # Display logo at the top of sidebar
    try:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown('<div style="text-align: center; font-size: 60px; padding: 20px;">🏥</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div style="text-align: center; font-size: 60px; padding: 20px;">🏥</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("⚙️ تنظیمات")
    
    # Role selection
    role = st.selectbox(
        "نقش خود را انتخاب کنید:",
        ["-- انتخاب کنید --", "بیمار", "پزشک"],
        key="role_selector"
    )
    
    st.markdown("---")
    
    # Show statistics
    st.subheader("📊 آمار جلسه")
    st.metric("تعداد مشاورات", len(st.session_state.consultation_history))
    
    if st.session_state.patient_symptoms:
        st.success("✅ علائم ثبت شده")
    else:
        st.warning("⏳ در انتظار ثبت علائم")
    
    st.markdown("---")
    
    # Emergency button
    if st.button("🚨 اورژانس 115", type="primary"):
        st.error("### تماس فوری با اورژانس:")
        st.info("📞 115")
    
    # Clear history
    if st.button("🗑️ پاک کردن تاریخچه"):
        st.session_state.consultation_history = []
        st.session_state.patient_symptoms = None
        st.session_state.analysis_result = None
        st.success("تاریخچه پاک شد!")
        st.rerun()

# Main content area
st.markdown("---")

# ============================================
# PATIENT INTERFACE
# ============================================

if role == "بیمار":
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tab selection for input method
        input_method = st.radio(
            "روش ورودی را انتخاب کنید:",
            ["🎙️ ورودی صوتی", "⌨️ ورودی متنی"],
            horizontal=True,
            key="input_method"
        )
        
        st.markdown("---")
        
        if input_method == "🎙️ ورودی صوتی":
            st.subheader("🎙️ ثبت علائم (ورودی صوتی)")

            # Voice recording workflow states
            if "is_recording" not in st.session_state:
                st.session_state.is_recording = False
            if "audio_data" not in st.session_state:
                st.session_state.audio_data = None

            fs = 44100  # Sample rate

            if not st.session_state.is_recording:
                # Show the "Start Recording" button when not recording
                if st.button("🎤 شروع ضبط صدا", type="primary"):
                    st.session_state.is_recording = True
                    st.session_state.audio_data = None
                    st.rerun()
            else:
                # Show the "Stop Recording" button while recording
                status_text = st.info("🔴 در حال ضبط... برای اتمام ضبط روی دکمه زیر کلیک کنید")

                # Start recording when entering recording mode
                if "recording_started" not in st.session_state:
                    try:
                        st.session_state.audio_array = sd.rec(int(60 * fs), samplerate=fs, channels=1)
                        st.session_state.recording_started = True
                    except Exception as e:
                        st.session_state.audio_array = None
                        st.session_state.is_recording = False
                        st.error(f"❌ خطا در ضبط صدا: {e}")
                        st.rerun()
                
                if st.button("⏹️ پایان ضبط"):
                    try:
                        sd.stop()
                        # Estimate length by finding number of samples before all zeros
                        # (Simple truncate extra silence if recording wasn't full 60s)
                        audio = st.session_state.audio_array
                        flat = audio.flatten()
                        # Find where actual input stops (optional for advanced users)
                        nonzero = (flat != 0).nonzero()[0]
                        if len(nonzero) > 0:
                            last_idx = nonzero[-1] + 1
                            audio = audio[:last_idx]
                        st.session_state.audio_data = audio
                        st.success("✅ ضبط تمام شد!")
                    except Exception as e:
                        st.session_state.audio_data = None
                        st.error(f"❌ خطا در ضبط صدا: {e}")
                    st.session_state.is_recording = False
                    if "recording_started" in st.session_state:
                        del st.session_state.recording_started
                    if "audio_array" in st.session_state:
                        del st.session_state.audio_array
                    st.rerun()

            # If audio recorded, proceed to saving and processing
            if st.session_state.audio_data is not None:
                audio = st.session_state.audio_data
                wavio.write("user_voice.wav", audio, fs, sampwidth=2)
                st.audio("user_voice.wav", format="audio/wav")
                
                # Save audio file
                wavio.write("user_voice.wav", audio, fs, sampwidth=2)
                st.audio("user_voice.wav", format="audio/wav")
                
                # Transcribe audio
                try:
                    with st.spinner("🔄 در حال تبدیل صوت به متن..."):
                        with open("user_voice.wav", "rb") as f:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=f,
                                language="fa"
                            )
                        
                        text = transcript.text
                        st.session_state.patient_symptoms = text
                        
                        st.success(f"📝 **علائم استخراج شده:**\n\n{text}")
                        
                        # Analyze symptoms
                        with st.spinner("🔍 در حال تحلیل علائم..."):
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            prompt = format_prompt(
                                PATIENT_ANALYSIS_PROMPT,
                                symptoms=text,
                                timestamp=timestamp
                            )
                            
                            analysis = ask_model(prompt)
                            st.session_state.analysis_result = analysis
                            
                            # Save to history
                            save_consultation(text, analysis, "بیمار")
                            
                            st.markdown("### 📋 نتیجه تحلیل:")
                            st.info(analysis)
                            
                            # Check for emergency
                            if "🔴" in analysis or "بحرانی" in analysis or "فوری" in analysis.lower():
                                st.error("⚠️ **هشدار:** احتمال نیاز به مراجعه فوری!")
                                if st.button("📞 تماس با اورژانس 115"):
                                    st.error("لطفاً فوراً با شماره 115 تماس بگیرید")
                    
                except Exception as e:
                    st.error(f"❌ خطا در پردازش صوت: {e}")
        
        elif input_method == "⌨️ ورودی متنی":
            st.subheader("⌨️ ثبت علائم (ورودی متنی)")
            
            # Text input form
            text_input = st.text_area(
                "علائم خود را به صورت متنی وارد کنید:",
                height=150,
                placeholder="مثال: من از دیروز صبح سردرد شدیدی دارم که از طرف راس سرم شروع شده. همراه با تهوع و حساسیت به نور. درد مثل ضربان است.",
                key="symptoms_text_input"
            )
            
            # Submit button
            if st.button("📤 ارسال و تحلیل علائم", type="primary"):
                if text_input.strip():
                    text = text_input.strip()
                    st.session_state.patient_symptoms = text
                    
                    st.success(f"📝 **علائم ثبت شده:**\n\n{text}")
                    
                    # Analyze symptoms
                    with st.spinner("🔍 در حال تحلیل علائم..."):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        prompt = format_prompt(
                            PATIENT_ANALYSIS_PROMPT,
                            symptoms=text,
                            timestamp=timestamp
                        )
                        
                        analysis = ask_model(prompt)
                        st.session_state.analysis_result = analysis
                        
                        # Save to history
                        save_consultation(text, analysis, "بیمار")
                        
                        st.markdown("### 📋 نتیجه تحلیل:")
                        st.info(analysis)
                        
                        # Check for emergency
                        if "🔴" in analysis or "بحرانی" in analysis or "فوری" in analysis.lower():
                            st.error("⚠️ **هشدار:** احتمال نیاز به مراجعه فوری!")
                            if st.button("📞 تماس با اورژانس 115", key="emergency_text"):
                                st.error("لطفاً فوراً با شماره 115 تماس بگیرید")
                else:
                    st.warning("⚠️ لطفاً علائم خود را وارد کنید")
    
    with col2:
        st.subheader("📌 راهنمای سریع")
        
        with st.expander("💡 چگونه علائم را بیان کنم؟"):
            st.markdown("""
            **نکات مهم:**
            - علائم خود را واضح و کامل توضیح دهید
            - زمان شروع را ذکر کنید
            - شدت درد را بگویید (خفیف/متوسط/شدید)
            - علائم همراه را نیز بگویید
            
            **مثال خوب:**
            "من از دیروز صبح سردرد شدیدی دارم که از طرف راس سرم شروع شده. همراه با تهوع و حساسیت به نور. درد مثل ضربان است."
            """)
        
        with st.expander("🚨 علائم خطرناک"):
            st.markdown("""
            **فوراً به اورژانس مراجعه کنید:**
            - درد قفسه سینه
            - تنگی نفس شدید
            - فلج یا ضعف ناگهانی
            - سردرد شدید و ناگهانی
            - خونریزی شدید
            - تب بالای 40 درجه
            - اختلال هوشیاری
            """)
        
        with st.expander("ℹ️ درباره این سیستم"):
            st.markdown("""
            این سیستم با استفاده از هوش مصنوعی پیشرفته:
            - علائم شما را تحلیل می‌کند
            - احتمال بیماری‌ها را بررسی می‌کند
            - توصیه‌های اولیه ارائه می‌دهد
            
            ⚠️ **توجه:** این سیستم جایگزین ویزیت پزشک نیست.
            """)

# ============================================
# DOCTOR INTERFACE
# ============================================

elif role == "پزشک":
    
    st.subheader("👨‍⚕️ پنل پزشک - تحلیل و پیگیری")
    
    if st.session_state.patient_symptoms is None:
        st.warning("⏳ در انتظار ثبت علائم توسط بیمار")
        st.info("لطفاً ابتدا بیمار باید علائم خود را در بخش مربوطه ثبت کند.")
    else:
        # Display patient symptoms
        st.success(f"✅ **علائم ثبت شده بیمار:**\n\n{st.session_state.patient_symptoms}")
        
        # Show initial analysis if available
        if st.session_state.analysis_result:
            with st.expander("📊 مشاهده تحلیل اولیه", expanded=False):
                st.info(st.session_state.analysis_result)
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("❓ پیشنهاد سوالات تکمیلی", type="primary"):
                with st.spinner("🔍 در حال تولید سوالات..."):
                    prompt = format_prompt(
                        DOCTOR_QUESTIONS_PROMPT,
                        symptoms=st.session_state.patient_symptoms
                    )
                    questions = ask_model(prompt)
                    st.markdown("### 📝 سوالات تکمیلی برای بیمار:")
                    st.info(questions)
        
        with col2:
            if st.button("🚨 بررسی فوریت"):
                with st.spinner("⚠️ در حال ارزیابی فوریت..."):
                    prompt = format_prompt(
                        EMERGENCY_PROTOCOL,
                        symptoms=st.session_state.patient_symptoms
                    )
                    emergency = ask_model(prompt)
                    st.markdown("### 🚨 ارزیابی فوریت:")
                    st.warning(emergency)
        
        with col3:
            if st.button("📄 تولید گزارش کامل"):
                st.info("🔄 در حال تولید گزارش جامع...")
                # This would generate a comprehensive report
                st.success("✅ گزارش در دست تهیه است...")

# ============================================
# CONSULTATION HISTORY
# ============================================

if st.session_state.consultation_history:
    st.markdown("---")
    st.subheader("📚 تاریخچه مشاورات")
    
    for i, consultation in enumerate(reversed(st.session_state.consultation_history)):
        with st.expander(f"مشاوره #{len(st.session_state.consultation_history) - i} - {consultation['timestamp']}"):
            st.markdown(f"**نقش:** {consultation['role']}")
            st.markdown(f"**علائم:** {consultation['symptoms']}")
            st.markdown("**تحلیل:**")
            st.info(consultation['analysis'])

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🏥 <strong>سیستم هوشمند مشاوره پزشکی</strong></p>
    <p>⚠️ این سیستم فقط برای مشاوره اولیه است و جایگزین ویزیت پزشک نمی‌شود</p>
    <p>🔒 اطلاعات شما محرمانه و امن نگهداری می‌شود</p>
</div>
""", unsafe_allow_html=True)