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
        <p>Basic demo interface for the education system dataset project.</p>
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
        st.error("Prediction failed. Please reload the app.")
    else:
        prediction = (
            coeffs[0]
            + coeffs[1] * attendance
            + coeffs[2] * study_hours
            + coeffs[3] * previous_score
        )
        st.success(f"Predicted Score: {prediction:.1f}%")
        st.caption(
            f"Trained weights: bias={coeffs[0]:.2f}, "
            f"attendance={coeffs[1]:.2f}, study_hours={coeffs[2]:.2f}, "
            f"previous_score={coeffs[3]:.2f}"
        )
st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    st.markdown("<div class='card'><h2>Dataset Details</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        This project fills the gap by developing an inclusive prediction and instructional
        support system. The student-related data includes:
        - % of Attendance
        - % of Hours of Study
        - Previous Scores
        """,
    )
    st.markdown("The dataset has approximately 100-200 total records.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>Data Preprocessing</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        - Elimination of gaps in data entry
        - Cleaning of data
        - Standardization of features
        """,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>Feature Engineering</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        - Hours Spent Studying - effort exerted
        - Attendance Rate - amount of class attendance
        - Past Test Scores - periodic means with which to measure progress
        """,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><h3>Train/Test Split</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        - 80% Training
        - 20% Testing
        """,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h2>System Architecture Diagram</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow">
        User Input<br/>
        ↓<br/>
        Flask Web Interface<br/>
        ↓<br/>
        Data Processing<br/>
        ↓<br/>
        Machine Learning Model (Linear Regression)<br/>
        ↓<br/>
        Predicted Output<br/>
        ↓<br/>
        Displayed to User
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h2>Implementation</h2>", unsafe_allow_html=True)
    st.markdown("This basic app uses Streamlit with a single page and simple styling.")
    st.markdown("</div>", unsafe_allow_html=True)
