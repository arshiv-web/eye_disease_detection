import gradio as gr
import integration
import pickle
from typing import Optional, List, Tuple

# --- Class Loading Function ---
def get_class_names() -> List[str]:
    CLASS_MAP_PATH = 'class2label.pk1' 
    with open(CLASS_MAP_PATH, 'rb') as f:
        class2label = pickle.load(f)
    
    # Sort the class names (keys) by their corresponding index (value)
    sorted_classes = sorted(class2label.keys(), key=lambda k: class2label[k])
    
    print(f"Loaded classes successfully: {sorted_classes}")
    return sorted_classes


# --- Constants ---
CLASSES = get_class_names()
NUM_CLASSES = len(CLASSES)


# --- UI Helper Functions ---

def stage_image(filepath: str, is_random: bool) -> tuple:
    try:
        if is_random:
            filepath = integration.get_random_study_image()
        
        staged_path = integration.prepare_stage(filepath)
        
        message = "Image staged. Please review and click **'Run Diagnosis'** to proceed."
        
        button_updates = [gr.update(visible=False) for _ in range(NUM_CLASSES)]
        
        return (
            staged_path,                            # 0: staged_file_state
            staged_path,                            # 1: display_image
            gr.update(visible=True),                # 2: show predict_btn
            gr.update(visible=True, value=message), # 3: show status_label
            gr.update(visible=False),               # 4: hide input_col
            gr.update(visible=False),               # 5: hide upload_image
            gr.update(visible=False),               # 6: hide random_btn
            *button_updates,                        # 7+: Update all 5 class buttons (hidden)
            gr.update(visible=False),               # N: hide prediction_output_row
            gr.update(visible=False),               # N+1: hide accept_btn
            gr.update(visible=False),               # N+2: hide reject_btn
        )
    except Exception as e:
        print(f"Error staging image: {e}")
        error_message = f"Error: {e}. Could not stage image."
        button_updates_error = [gr.update(visible=False) for _ in range(NUM_CLASSES)]
        return (
            None, None,
            gr.update(visible=False),
            gr.update(visible=True, value=error_message),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), # Show inputs
            *button_updates_error,
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        )


def predict_staged_image(staged_path: Optional[str]) -> tuple:
    if not staged_path:
        return reset_ui_components(message="Error: No image staged. Please select an image first.")

    try:
        predictions: List[Tuple[str, float]] = integration.predict_single_image() 
        
        button_updates = []
        for i, (cls, prob) in enumerate(predictions[:NUM_CLASSES]):
            if i == 0:
                display_text = f"{cls} ({prob*100:.1f}%)"
                variant = "success"
            else:
                display_text = f"{cls} ({prob*100:.1f}%)"
                variant = "secondary"
            
            button_updates.append(
                gr.update(value=display_text, variant=variant, visible=True)
            )
        
        while len(button_updates) < NUM_CLASSES:
             button_updates.append(gr.update(visible=False))

        status_message = "Diagnosis complete. Review the results below and provide **3. Feedback**."
        return (
            staged_path,                          # 0: staged_file_state
            staged_path,                          # 1: display_image
            gr.update(visible=False),             # 2: predict_btn (hide)
            gr.update(visible=True, value=status_message), # 3: status_label
            gr.update(visible=False),             # 4: input_col (hide)
            gr.update(visible=False),             # 5: upload_image (hide)
            gr.update(visible=False),             # 6: random_btn (hide)
            *button_updates,                      # 7+: Update all 5 class buttons
            gr.update(visible=True),              # N: show prediction_output_row
            gr.update(visible=True),              # N+1: accept_btn (show)
            gr.update(visible=True),              # N+2: reject_btn (show)
        )
    except Exception as e:
        print(f"Error running prediction: {e}")
        error_message = f"Prediction Error: {e}. Please check logs, ensure the image is valid, and try again."
        return reset_ui_components(message=error_message)


def handle_feedback_and_reset(feedback_type: str) -> tuple:
    confirmation_message = integration.handle_feedback(feedback_type)
    
    reset_outputs = reset_ui_components()
    
    output_list = list(reset_outputs)
    output_list[3] = gr.update(visible=True, value=confirmation_message) 
    
    return tuple(output_list)


def reset_ui_components(message: str = None) -> tuple:
    default_message = "**Welcome!** Upload an eye image or select a random one to start."
    initial_message = message if message else default_message
    
    button_resets = [gr.update(visible=False) for _ in range(NUM_CLASSES)]
    
    return (
        None,                       # 0: Clear staged_file_state
        None,                       # 1: Clear display_image
        gr.update(visible=False),   # 2: Hide predict_btn
        gr.update(visible=True, value=initial_message),    # 3: Show status_label
        gr.update(visible=True),    # 4: Show input_col
        gr.update(visible=True),    # 5: Show upload_image
        gr.update(visible=True),    # 6: Show random_btn
        *button_resets,             # 7+: Reset the class buttons (hidden)
        gr.update(visible=False),   # N: Hide prediction_output_row
        gr.update(visible=False),   # N+1: Hide accept_btn
        gr.update(visible=False),   # N+2: Hide reject_btn
    )


def reset_ui() -> tuple:
    """
    Clears the stage and returns the UI to its default state (used for the Reset button).
    """
    integration.clear_stage()
    return reset_ui_components()

# --- Build the Gradio UI ---

INITIAL_MESSAGE = "**Welcome!** Upload an eye image or select a random one to start."

with gr.Blocks(theme=gr.themes.Soft(), title="Eye Diagnosis Assistant") as demo:
    gr.Markdown("# 🩺 Eye Disease Diagnosis Assistant")
    gr.Markdown(
        "Upload an image or get a random one. **Images are resized for consistent model input.**"
    )
    
    staged_file_state = gr.State(value=None)
    status_label = gr.Markdown(INITIAL_MESSAGE)

    with gr.Row():
        # --- Column 1: Inputs & Controls ---
        with gr.Column(scale=1):
            with gr.Column(visible=True) as input_col:
                gr.Markdown("### 1. Select Image")
                upload_image = gr.Image(
                    type="filepath", 
                    label="Upload an Eye Image",
                    height=250,
                    visible=True 
                )
                gr.Markdown("<center>— OR —</center>")
                random_btn = gr.Button("👁️ Get Random Image from Study Set", visible=True)

            predict_btn = gr.Button(
                "🔬 Run Diagnosis", 
                variant="primary", 
                visible=False 
            )
            gr.Markdown("---")
            reset_btn = gr.Button("🔄 Reset / Start Over")

        # --- Column 2: Outputs & Feedback ---
        with gr.Column(scale=2):
            # Display size set to 350x350
            display_image = gr.Image(
                label="Image for Review",
                height=350,
                width=350,
                image_mode='RGB',
                interactive=False 
            )
            
            # Prediction Output Section
            with gr.Column(visible=False) as prediction_output_row:
                gr.Markdown("### Diagnosis Results & Feedback")
                gr.Markdown(f'Probabilities are shown for all classes.')
                
                # We need 5 distinct output components for the 5 classes
                class_buttons = []
                for i in range(NUM_CLASSES):
                    btn = gr.Button(
                        value=f"Class {i+1}", 
                        scale=1,
                        variant="secondary", 
                        visible=False,
                        interactive=False 
                    )
                    class_buttons.append(btn)
                
                # --- MODIFIED: Top Prediction in its own row and is larger (scale=2) ---
                
                # Row 1: Top Prediction (Index 0)
                with gr.Row():
                    if NUM_CLASSES > 0:
                        top_pred_btn = class_buttons[0]
                        # Set scale to 2 to make it twice as wide as the secondary buttons
                        top_pred_btn.scale = 2 
                    else:
                        top_pred_btn = gr.Markdown("Error: No classes defined.")

                # Row 2: Remaining Predictions (Index 1 to 4)
                with gr.Row():
                    if NUM_CLASSES > 1:
                        # The remaining 4 buttons are displayed here, using default scale=1
                        for btn in class_buttons[1:]:
                            pass # Components are already created, no need to recreate them
                
                # --- END MODIFIED ---
                
                # Feedback buttons 
                with gr.Row():
                    accept_btn = gr.Button(
                        "✅ Accept Diagnosis & Log Feedback", 
                        variant="primary", 
                        visible=False
                    )
                    reject_btn = gr.Button(
                        "❌ Reject Diagnosis & Log Feedback", 
                        variant="stop", 
                        visible=False
                    )


    # --- Define Component Interactions ---
    
    class_button_outputs = class_buttons 
    
    # This list must EXACTLY match the return order of all UI helper functions
    full_outputs = [
        staged_file_state, display_image, predict_btn, status_label, input_col, upload_image, random_btn, 
        *class_button_outputs, prediction_output_row, accept_btn, reject_btn
    ]

    upload_image.upload(fn=lambda x: stage_image(x, is_random=False), inputs=[upload_image], outputs=full_outputs, show_progress='full')
    random_btn.click(fn=lambda: stage_image(None, is_random=True), inputs=None, outputs=full_outputs, show_progress='full')
    
    predict_btn.click(fn=predict_staged_image, inputs=[staged_file_state], outputs=full_outputs, show_progress='full')

    accept_btn.click(fn=lambda: handle_feedback_and_reset("accepted"), inputs=None, outputs=full_outputs)
    reject_btn.click(fn=lambda: handle_feedback_and_reset("rejected"), inputs=None, outputs=full_outputs)
    reset_btn.click(fn=reset_ui, inputs=None, outputs=full_outputs)
    
    demo.load(fn=reset_ui, inputs=None, outputs=full_outputs)


# --- How to Run ---
if __name__ == "__main__":
    print("Launching Gradio Interface...")
    print("Access it at: http://127.0.0.1:7860")
    demo.launch(share=False)