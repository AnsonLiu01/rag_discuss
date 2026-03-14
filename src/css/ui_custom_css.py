import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #f6f8fa; }

        [data-testid="stChatMessage"] {
            background-color: white !important;
            border-radius: 12px !important;
            padding: 16px !important;
            margin-bottom: 8px !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
            border: 1px solid #edf2f7 !important;
        }

        [data-testid="stChatMessageAvatar"] { display: none !important; }

        .sender-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
            color: #718096;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
        }

        .label-marker {
            width: 4px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
        }

        [data-testid="stChatMessage"][data-test-role="assistant"] .sender-label .label-marker {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        [data-testid="stChatMessage"][data-test-role="user"] .sender-label .label-marker {
            background: linear-gradient(135deg, #1a202c 0%, #4a5568 100%);
        }

        .block-container { padding-top: 2rem !important; }

        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0;
        }

        h1 { font-weight: 900 !important; letter-spacing: -1px; color: #1a202c; }

        /* Source Inspector Styling */
        .confidence-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
            margin: 12px 0;
        }

        .confidence-high {
            background: #d1fae5;
            color: #065f46;
        }

        .confidence-medium {
            background: #fef3c7;
            color: #92400e;
        }

        .confidence-low {
            background: #fee2e2;
            color: #991b1b;
        }

        .source-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            margin-top: 12px;
        }

        .source-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .source-name {
            font-size: 11px;
            font-weight: 600;
            color: #334155;
            font-family: 'Monaco', 'Courier New', monospace;
            background: #ffffff;
            padding: 6px 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            word-break: break-all;
        }

        .content-preview {
            background: #ffffff;
            border-left: 3px solid #667eea;
            padding: 10px;
            border-radius: 4px;
            font-size: 11px;
            line-height: 1.5;
            color: #475569;
            max-height: 150px;
            overflow-y: auto;
        }

        .empty-state {
            text-align: center;
            padding: 20px;
            color: #94a3b8;
        }

        .empty-icon {
            font-size: 32px;
            margin-bottom: 8px;
            opacity: 0.5;
        }

        .empty-text {
            font-size: 11px;
            line-height: 1.4;
        }
        </style>
    """, unsafe_allow_html=True)