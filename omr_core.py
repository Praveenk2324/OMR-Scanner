import gradio as gr
import cv2
import numpy as np
import imutils
import os
import re
import pandas as pd

def parse_answer_key(key_file_path):
    answers = {}
    try:
        df = pd.read_csv(key_file_path, header=None)
        for item in df.values.flatten():
            if pd.isna(item):
                continue
            match = re.search(r'(\d+)\s*[-.\s]\s*([a-d])', str(item), re.IGNORECASE)
            if match:
                q_num = int(match.group(1))
                answer_letter = match.group(2).upper()
                answers[q_num] = answer_letter
    except Exception as e:
        return {"error": f"Failed to parse answer key: {e}"}
    return answers

def find_and_sort_bubbles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    question_cnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.8 and ar <= 1.3:
            question_cnts.append(c)

    question_cnts = sorted(question_cnts, key=lambda c: (cv2.boundingRect(c)[0], cv2.boundingRect(c)[1]))

    if len(question_cnts) > 100:
        x_coords = [cv2.boundingRect(c)[0] for c in question_cnts]
        col_width = (max(x_coords) - min(x_coords)) / 5
        
        def get_col(x):
            return min(4, int((x - min(x_coords)) / col_width))

        question_cnts = sorted(question_cnts, key=lambda c: (get_col(cv2.boundingRect(c)[0]), cv2.boundingRect(c)[1]))

    return question_cnts, thresh

def extract_student_answers(question_cnts, thresh):
    student_answers = {}
    num_questions = len(question_cnts) // 4
    for i in range(num_questions):
        start_index = i * 4
        end_index = start_index + 4
        
        row_cnts = sorted(question_cnts[start_index:end_index], key=lambda c: cv2.boundingRect(c)[0])
        
        bubbled = None
        for j, c in enumerate(row_cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        
        if bubbled and bubbled[0] > 150:
            student_answers[i + 1] = chr(ord('A') + bubbled[1])
            
    return student_answers

def score_omr_sheets(omr_images, answer_key_file):
    if omr_images is None or answer_key_file is None:
        return "<p style='color: red; text-align: center;'>Please upload both OMR sheets and an answer key file.</p>"

    answer_key = parse_answer_key(answer_key_file.name)
    if "error" in answer_key:
        return f"<p style='color: red; text-align: center;'>{answer_key['error']}</p>"

    final_html_report = ""

    for image_obj in omr_images:
        try:
            img = cv2.imread(image_obj.name)
            filename = os.path.basename(image_obj.name)

            bubbles, thresh = find_and_sort_bubbles(img)
            
            if len(bubbles) < 100:
                final_html_report += f"<h2>Results for {filename}</h2><p style='color: orange;'>Warning: Could not detect enough bubbles ({len(bubbles)} found). The sheet may be skewed or of poor quality. Skipping.</p><hr>"
                continue

            student_answers = extract_student_answers(bubbles, thresh)

            score = 0
            results_data = []
            for q_num in range(1, 101):
                correct_answer = answer_key.get(q_num, "N/A")
                student_answer = student_answers.get(q_num, "Unanswered")
                status = "Unanswered"
                
                if student_answer != "Unanswered":
                    if student_answer == correct_answer:
                        status = "Correct"
                        score += 1
                    else:
                        status = "Incorrect"
                results_data.append([q_num, student_answer, correct_answer, status])
            
            final_html_report += f"<h2>Results for {filename}</h2>"
            final_html_report += f"<p><b>Score: {score} / 100 ({score}%)</b></p>"
            
            df = pd.DataFrame(results_data, columns=["Question #", "Your Answer", "Correct Answer", "Status"])
            
            def style_rows(row):
                if row['Status'] == 'Correct':
                    return ['background-color: #d4edda'] * len(row)
                elif row['Status'] == 'Incorrect':
                    return ['background-color: #f8d7da'] * len(row)
                return [''] * len(row)
            
            styled_df = df.style.apply(style_rows, axis=1).set_table_attributes('border="1" class="dataframe"')
            final_html_report += styled_df.to_html()
            final_html_report += "<hr>"

        except Exception as e:
            final_html_report += f"<h2>Error processing {os.path.basename(image_obj.name)}</h2><p style='color: red;'>An unexpected error occurred: {e}</p><hr>"
    
    return final_html_report

with gr.Blocks(theme=gr.themes.Soft(), title="OMR Scoring System") as demo:
    gr.Markdown("# Automated OMR Scoring System")
    gr.Markdown("Upload one or more OMR sheets and a single answer key file (`.csv`) to automatically score them.")
    
    with gr.Row():
        with gr.Column(scale=1):
            omr_sheets_input = gr.File(
                label="Upload OMR Sheets",
                file_count="multiple",
                file_types=["image"]
            )
            answer_key_input = gr.File(
                label="Upload Answer Key (CSV)",
                file_count="single",
                file_types=[".csv"]
            )
            process_button = gr.Button("Start Scoring", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Scoring Results")
            results_output = gr.HTML("Your results will appear here...")

    process_button.click(
        fn=score_omr_sheets,
        inputs=[omr_sheets_input, answer_key_input],
        outputs=results_output
    )

if __name__ == "__main__":
    demo.launch()