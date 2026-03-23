import streamlit as st


st.set_page_config(page_title="Education Platform", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f9fc, #e6eef6);
        color: #1e293b;
    }
    .main {
        background: linear-gradient(135deg, #f7f9fc, #e6eef6);
    }
    .hero {
        background: #ffffff;
        padding: 24px 28px;
        border-radius: 18px;
        box-shadow: 0 16px 40px rgba(18, 22, 26, 0.15);
        margin-bottom: 24px;
        color: #1e293b;
    }
    .card {
        background: #ffffff;
        padding: 20px 24px;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(18, 22, 26, 0.12);
        margin-bottom: 18px;
        color: #1e293b;
    }
    .result-card {
        background: #eaf2ff;
        border: 1px solid #c7dbff;
        padding: 16px 18px;
        border-radius: 14px;
        margin-top: 16px;
        color: #0f172a;
        font-weight: 600;
    }
    .module-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .module-tile {
        background: #f1f5f9;
        border: 1px solid #d4e1f5;
        padding: 16px 18px;
        border-radius: 16px;
        font-weight: 600;
        color: #1e293b;
        text-align: center;
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.2);
    }
    .flow {
        font-weight: 600;
        line-height: 1.8;
    }
    h1, h2, h3, p, li, span, label {
        color: #1e293b;
    }
    .stMarkdown p {
        color: #334155;
    }
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: #1e293b !important;
    }
    .stButton>button {
        background: #1d4ed8;
        color: #ffffff;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Education Platform</h1>
        <p>Choose a module to get started.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">
        <h2>Dashboard Modules</h2>
        <div class="module-grid">
            <div class="module-tile">Exam Predictor</div>
            <div class="module-tile">Smart Tutor</div>
            <div class="module-tile">Essay Grader</div>
            <div class="module-tile">Attendance</div>
            <div class="module-tile">QA Bot</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def generate_dataset():
    import random

    rng = random.Random(42)
    features = []
    targets = []
    for _ in range(150):
        attendance = rng.uniform(60, 100)
        study_hours = rng.uniform(40, 100)
        previous_score = rng.uniform(45, 98)
        noise = rng.gauss(0, 3)
        predicted = (
            0.4 * attendance + 0.3 * study_hours + 0.3 * previous_score + noise
        )
        features.append([attendance, study_hours, previous_score])
        targets.append(predicted)
    return features, targets


def solve_linear_system(matrix, vector):
    n = len(vector)
    for pivot in range(n):
        max_row = max(range(pivot, n), key=lambda r: abs(matrix[r][pivot]))
        if abs(matrix[max_row][pivot]) < 1e-12:
            raise ValueError("Singular matrix encountered during training.")
        if max_row != pivot:
            matrix[pivot], matrix[max_row] = matrix[max_row], matrix[pivot]
            vector[pivot], vector[max_row] = vector[max_row], vector[pivot]

        pivot_val = matrix[pivot][pivot]
        for col in range(pivot, n):
            matrix[pivot][col] /= pivot_val
        vector[pivot] /= pivot_val

        for row in range(n):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            for col in range(pivot, n):
                matrix[row][col] -= factor * matrix[pivot][col]
            vector[row] -= factor * vector[pivot]
    return vector


def train_linear_regression(features, targets):
    x_rows = [[1.0] + row for row in features]
    cols = len(x_rows[0])
    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [0.0 for _ in range(cols)]

    for row, y in zip(x_rows, targets):
        for i in range(cols):
            xty[i] += row[i] * y
            for j in range(cols):
                xtx[i][j] += row[i] * row[j]

    return solve_linear_system(xtx, xty)


st.markdown("<div class='card'><h2>Quick Prediction</h2>", unsafe_allow_html=True)

@st.cache_data
def get_coeffs():
    features, targets = generate_dataset()
    split_index = int(len(features) * 0.8)
    return train_linear_regression(features[:split_index], targets[:split_index])

coeffs = None
training_error = None
try:
    coeffs = get_coeffs()
except Exception as exc:
    training_error = str(exc)

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
    with col2:
        study_hours = st.number_input("Hours of Study (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    with col3:
        previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
    submitted = st.form_submit_button("Predict Score")

if submitted:
    if training_error or coeffs is None:
        st.markdown(
            "<div class='result-card'>Prediction failed. Please reload the app.</div>",
            unsafe_allow_html=True,
        )
    else:
        prediction = (
            coeffs[0]
            + coeffs[1] * attendance
            + coeffs[2] * study_hours
            + coeffs[3] * previous_score
        )
        st.markdown(
            f"<div class='result-card'>Predicted Score: {prediction:.1f}%</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Trained weights: bias={coeffs[0]:.2f}, "
            f"attendance={coeffs[1]:.2f}, study_hours={coeffs[2]:.2f}, "
            f"previous_score={coeffs[3]:.2f}"
        )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="card">
        <h2>Smart Tutor</h2>
        <p>Get a quick study tip for a topic.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.form("smart_tutor_form"):
    topic = st.text_input("Topic for Smart Tutor", key="smart_tutor_topic")
    smart_submit = st.form_submit_button("Generate Tip")
if smart_submit:
    if topic.strip():
        st.info(f"Tip: Break '{topic.strip()}' into small concepts and practice with 3 short questions.")
    else:
        st.warning("Please enter a topic.")

st.markdown(
    """
    <div class="card">
        <h2>Essay Grader</h2>
        <p>Paste an essay draft to get quick feedback.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.form("essay_grader_form"):
    essay = st.text_area("Essay Draft", height=160, key="essay_draft")
    essay_submit = st.form_submit_button("Grade Essay")
if essay_submit:
    if essay.strip():
        st.success("Feedback: Clear structure. Add two specific examples and tighten the conclusion.")
    else:
        st.warning("Please paste an essay draft.")

st.markdown(
    """
    <div class="card">
        <h2>Attendance</h2>
        <p>Record attendance quickly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.form("attendance_form"):
    att_col1, att_col2 = st.columns(2)
    with att_col1:
        roll_no = st.text_input("Roll No", key="attendance_roll_no")
    with att_col2:
        student_name = st.text_input("Student Name", key="attendance_student_name")
    slot_col1, slot_col2 = st.columns(2)
    with slot_col1:
        class_slot = st.selectbox("Class Slot", ["Morning", "Afternoon", "Evening"], key="attendance_class_slot")
    with slot_col2:
        status = st.selectbox("Status", ["Present", "Absent", "Late"], key="attendance_status")
    attendance_submit = st.form_submit_button("Submit Attendance")
if attendance_submit:
    if roll_no.strip() and student_name.strip():
        st.success(f"Attendance saved for {student_name.strip()} ({roll_no.strip()}) - {status}.")
    else:
        st.warning("Please enter roll number and student name.")

st.markdown(
    """
    <div class="card">
        <h2>QA Bot</h2>
        <p>Ask a question about your studies.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.form("qa_bot_form"):
    question = st.text_area("Ask a question", key="qa_bot_question")
    qa_submit = st.form_submit_button("Get Answer")
if qa_submit:
    if question.strip():
        st.info("Answer: Focus on key definitions first, then solve a small example.")
    else:
        st.warning("Please enter a question.")
