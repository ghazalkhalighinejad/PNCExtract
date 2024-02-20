from munkres import Munkres

def match_samples(is_js_scores, len_i, len_j):

    matrix = [[0 for j in range(len_j)] for i in range(len_i)]

    
    for i_j_score in is_js_scores:
        i = i_j_score[0]
        j = i_j_score[1]
        score = i_j_score[2]
        matrix[i][j] = -1 * float(score)
    
    m = Munkres()
    indexes = m.compute(matrix)
    matches = []
    for row, column in indexes:
        matches.append([row, column, matrix[row][column]*-1])
    return matches