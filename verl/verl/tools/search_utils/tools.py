from verl.tools.search_utils.utils import *
import time

# search
def search(queries, top_k=5):
    """
    Args:
        queries (list): 查询list
    """
    if len(queries) > 1:
        queries = queries[:1]

    results = []
    results_string = []
    for i in range(5):
        try:
            start_time = time.time()
            res = ddgs_search(queries, top_k, ddgs_backend="auto")
            end_time = time.time()
            print(f"google search query_list: {queries}, time taken: {end_time - start_time} seconds")
            if res:
                return generate_search_snippets(res)
            else:
                continue
        except Exception as e:
            continue

    return "Google search encountered an error and was unable to extract valid information."


if __name__ == '__main__':
    query_list = ['北京大学 简介']
    search_res = search(query_list, top_k=3)
    print(search_res)