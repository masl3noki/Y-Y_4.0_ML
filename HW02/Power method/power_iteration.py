import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    # Источник - википедия.
    # Сначала инициализируем стартовый вектор. Предлагается сгенерировать рандомно:
    r_k = np.random.rand(data.shape[0])
    for _ in range(num_steps):
        step = data @ r_k
        step_norm = np.linalg.norm(step)
        r_k = step / step_norm

    # Отл! Теперь находим собственные вектора. Известно, что Ax = lx, где l - собств. значение, а x - собств. вектор.
    # И тогда x.T @ A @ x = l. Только еще нужно разделить на норму x:

    e_val = (r_k.T @ data @ r_k) / np.linalg.norm(r_k)
    e_val = float(e_val)  # Просят перевести во float
    result = data - np.diag(np.full(data.shape[0], e_val))  # Это не нужно, однако это матрица A-l@E. Через нее мы получаем собств. вектора на бумаге

    return e_val, r_k
    
    