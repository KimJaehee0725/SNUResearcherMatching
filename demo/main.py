import gradio as gr
from utils import search_full_db, search_subset_db, find_similar_professors, get_model_and_corpus_with_index


def format_results_for_dataset(results):
    """
    Formats search results for display in a Gradio Dataset component.

    This function converts a dictionary of search results into a list of
    HTML-formatted strings, suitable for rendering in a Gradio Markdown component
    within a Dataset.

    Args:
        results (dict): A dictionary of search results, where keys are professor
                        names and values are dictionaries of their details.

    Returns:
        list: A list of lists, where each inner list contains a single
              HTML-formatted string representing a search result.
    """
    samples = []
    for prof_name, prof_info in results.items():
        # Using Markdown with HTML for better formatting
        
        docs_html_parts = []
        document_keys = [k for k in prof_info.keys() if 'document' in k]

        if document_keys:
            # Extract document numbers and find the max
            doc_numbers = [int(k.split('document')[1]) for k in document_keys]
            max_docs = max(doc_numbers) if doc_numbers else 0
            
            for i in range(1, max_docs + 1):
                doc_key = f'document{i}'
                type_key = f'type{i}'

                if doc_key in prof_info:
                    doc_type = prof_info.get(type_key, '')
                    
                    type_display = ""
                    if doc_type == 'research':
                        type_display = "<span style='color: #007bff; font-weight: 500;'>[연구]</span>"
                    elif doc_type == 'project':
                        type_display = "<span style='color: #28a745; font-weight: 500;'>[과제]</span>"

                    # Each document is in a div, with controlled padding to manage spacing
                    docs_html_parts.append(f"<div style='padding: 4px 0;'>📄 {type_display} {prof_info[doc_key]}</div>")

        docs_html = "".join(docs_html_parts)

        # Main container with card-like styling
        html_content = f"""
<div style='border: 1px solid #eaeaea; border-radius: 8px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); font-family: sans-serif;'>
    <div style='text-align: left;'>
        <b style='font-size: 1.2em;'>👤 {prof_name}</b><br>
        <span style='color:gray; font-size: 1.0em;'>🏢 {prof_info.get('department', '')}</span>
    </div>
    <hr style='margin: 12px 0; border: 0; border-top: 1px solid #eee;'>
    <div style='text-align: left; font-size: 1.0em; color: #333;'>
        {docs_html}
    </div>
</div>
"""
        samples.append([html_content])
    return samples

with gr.Blocks() as demo:
    # State to hold the full search results, allowing them to be accessed by click handlers
    full_db_results_state = gr.State()
    subset_db_results_state = gr.State()

    gr.Markdown("## 서울대학교 공과대학 연구자 매칭 시스템 데모")

    # 1. User query input
    with gr.Row():
        query_input = gr.Textbox(label="검색어 입력", placeholder="예: 인공지능, 반도체, 로봇", scale=4)
        k_input = gr.Number(label="교수님 수(k)", value=3, minimum=1, step=1)
        p_input = gr.Number(label="최대 문서 수(p)", value=5, minimum=1, step=1)
        search_button = gr.Button("검색", variant="primary", scale=1)

    # 2. Search results from two different databases
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 관련 교수님 검색 결과")
            full_db_dataset = gr.Dataset(
                components=[gr.Markdown()],
                samples=[],
                samples_per_page=5,
                label="관련 교수님 검색 결과",
                visible=False,
            )

        with gr.Column():
            gr.Markdown("### 신진 교수님 검색 결과")
            subset_db_dataset = gr.Dataset(
                components=[gr.Markdown()],
                samples=[],
                samples_per_page=5,
                label="신진 교수님 검색 결과",
                visible=False,
            )

    # 3. Displaying professors similar to the selected document
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 선택된 연구실과 관련된 교수님")
            similar_prof_dataset = gr.Dataset(
                components=[gr.Markdown()],
                samples=[],
                samples_per_page=3,
                label="선택된 연구실과 관련된 교수님",
                visible=False,
            )

    # --- Event Handlers ---

    def perform_main_search(query, k_full, num_docs_full, k_subset, num_docs_subset):
        """
        Event handler for the main search button click.

        Performs searches on both the full and subset databases and updates the UI
        with the results.

        Args:
            query (str): The search query entered by the user.
            k_full (int): The number of professors to retrieve from the full database.
            num_docs_full (int): The number of documents per professor from the full database.
            k_subset (int): The number of professors to retrieve from the subset database.
            num_docs_subset (int): The number of documents per professor from the subset database.

        Returns:
            tuple: A tuple of Gradio component updates for the datasets and states.
        """
        full_results = search_full_db(
            user_q      = query, 
            index       = full_index, 
            corpus      = full_corpus, 
            model       = model, 
            large_k     = large_k_full, 
            k_full      = k_full, 
            num_docs_full= num_docs_full
            )
        subset_results = search_subset_db(
            user_q      = query, 
            index       = subset_index, 
            corpus      = subset_corpus, 
            model       = model, 
            large_k     = large_k_subset, 
            k_subset    = k_subset, 
            num_docs_subset = num_docs_subset
            )

        full_db_samples = format_results_for_dataset(full_results)
        subset_db_samples = format_results_for_dataset(subset_results)

        # The return values update the UI components and the state variables
        return (
            gr.update(samples=full_db_samples, visible=True),
            gr.update(samples=subset_db_samples, visible=True),
            gr.update(samples=[], visible=False), # Hide similar profs dataset and clear content
            full_results,
            subset_results,
        )

    def find_and_show_similar(full_db_results, subset_db_results, selected_results_from_state, evt: gr.SelectData, k_similar, num_docs_similar):
        """
        Finds and displays professors similar to a selected document.

        This function is triggered when a user clicks on an item in either the
        full or subset result datasets. It uses the selected document to find
        other professors with similar work.

        Args:
            full_db_results (dict): The results from the full database search.
            subset_db_results (dict): The results from the subset database search.
            selected_results_from_state (dict): The state containing the results of the dataset that was clicked.
            evt (gr.SelectData): The event data from the selection, containing the index of the clicked item.
            k_similar (int): The number of similar professors to find.
            num_docs_similar (int): The number of documents to show for each similar professor.

        Returns:
            gr.update: A Gradio update for the similar professors dataset.
        """
        excluded_profs = set(full_db_results.keys()) | set(subset_db_results.keys())
        
        document_to_search = list(selected_results_from_state.values())[evt.index]['document1']

        similar_profs = find_similar_professors(
            document    = document_to_search, 
            index       = full_index, 
            corpus      = full_corpus, 
            model       = model, 
            large_k     = large_k_full, 
            k_similar   = k_similar, 
            num_docs_similar = num_docs_similar,
            excluded_profs=excluded_profs
            )

        if not similar_profs:
            return gr.update(samples=[], visible=False)
        
        # Format output for the Dataset component
        similar_prof_samples = format_results_for_dataset(similar_profs)
            
        return gr.update(samples=similar_prof_samples, visible=True)

    def on_select_full_db(evt: gr.SelectData, full_db_results, subset_db_results, k, p):
        """
        Event handler for selection in the full database results dataset.
        
        Args:
            evt (gr.SelectData): The event data from the selection.
            full_db_results (dict): The state holding results for the full DB.
            subset_db_results (dict): The state holding results for the subset DB.
            k (int): The number of similar professors to find.
            p (int): The number of documents per similar professor.

        Returns:
            The result of the find_and_show_similar function.
        """
        return find_and_show_similar(full_db_results, subset_db_results, full_db_results, evt, k, p)

    def on_select_subset_db(evt: gr.SelectData, full_db_results, subset_db_results, k, p):
        """
        Event handler for selection in the subset database results dataset.
        
        Args:
            evt (gr.SelectData): The event data from the selection.
            full_db_results (dict): The state holding results for the full DB.
            subset_db_results (dict): The state holding results for the subset DB.
            k (int): The number of similar professors to find.
            p (int): The number of documents per similar professor.

        Returns:
            The result of the find_and_show_similar function.
        """
        return find_and_show_similar(full_db_results, subset_db_results, subset_db_results, evt, k, p)

    # Wire the main search button to its handler function
    search_button.click(
        fn=perform_main_search,
        inputs=[query_input, k_input, p_input, k_input, p_input],
        outputs=[
            full_db_dataset,
            subset_db_dataset,
            similar_prof_dataset,
            full_db_results_state,
            subset_db_results_state,
        ],
    )

    # Wire the select event of each Dataset to the handler
    full_db_dataset.select(
        fn=on_select_full_db,
        inputs=[full_db_results_state, subset_db_results_state, k_input, p_input],
        outputs=[similar_prof_dataset],
    )
    subset_db_dataset.select(
        fn=on_select_subset_db,
        inputs=[full_db_results_state, subset_db_results_state, k_input, p_input],
        outputs=[similar_prof_dataset],
    )

if __name__ == "__main__":
    import argparse
    import os

    def str_to_bool(value):
        return str(value).lower() in {"1", "true", "yes", "y"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/workspace/trained_model/checkpoint-2060")
    parser.add_argument("--research_corpus_path", type=str, default="/workspace/data/translation_data/research_translation_final.csv")
    parser.add_argument("--project_corpus_path", type=str, default="/workspace/data/translation_data/project_translation_final.csv")
    parser.add_argument("--research_new_prof_path", type=str, default="/workspace/data/new_researchers/research_new_prof_10.txt")
    parser.add_argument("--project_new_prof_path", type=str, default="/workspace/data/new_researchers/project_new_prof_10.txt")
    parser.add_argument("--index_path", type=str, default="/workspace/data/demo/ver1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--server_name", type=str, default=os.environ.get("GRADIO_SERVER_NAME"))
    parser.add_argument("--server_port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", type=str_to_bool, default=str_to_bool(os.environ.get("GRADIO_SHARE", "true")))

    args = parser.parse_args()

    model, full_corpus, full_index, subset_corpus, subset_index = get_model_and_corpus_with_index(
        device=args.device,
        model_path=args.model_path,
        research_corpus_path=args.research_corpus_path,
        project_corpus_path=args.project_corpus_path,
        research_new_prof_path=args.research_new_prof_path,
        project_new_prof_path=args.project_new_prof_path,
        index_path=args.index_path
    )
    large_k_full = len(full_corpus)
    large_k_subset = len(subset_corpus)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
