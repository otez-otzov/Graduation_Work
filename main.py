import os
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import hashlib

app = Flask(__name__)
CORS(app)


# ----------------------------
# Граф и матрица переходов
# ----------------------------
def create_graph():
    G = nx.Graph()

    rooms = [
        'Entrance', 'Exit', 'Hallway', 'Shop1', 'Shop2', 'Shop3', 'Shop4', 'Shop5', 'Shop6',
        'FoodCourt', 'Toilets', 'Elevator', 'Stairs', 'Parking', 'Office1', 'Office2',
        'Kiosk1', 'Kiosk2', 'RestArea1', 'RestArea2', 'InfoDesk',
        'Lounge1', 'Lounge2', 'MiniShop1', 'MiniShop2'
    ]
    G.add_nodes_from(rooms)

    connections = [
        ('Entrance', 'Hallway'),
        ('Hallway', 'Shop1'),
        ('Hallway', 'Shop2'),
        ('Hallway', 'Shop3'),
        ('Shop3', 'Shop4'),
        ('Shop4', 'Shop5'),
        ('Shop5', 'Shop6'),
        ('Hallway', 'FoodCourt'),
        ('FoodCourt', 'Toilets'),
        ('Hallway', 'Elevator'),
        ('Elevator', 'Stairs'),
        ('Stairs', 'Parking'),
        ('Parking', 'Office1'),
        ('Office1', 'Office2'),
        ('Exit', 'Shop6'),
        ('Exit', 'Office2'),
        ('Toilets', 'Kiosk1'),
        ('Kiosk1', 'Kiosk2'),
        ('Kiosk2', 'RestArea1'),
        ('RestArea1', 'RestArea2'),
        ('RestArea2', 'InfoDesk'),
        ('InfoDesk', 'Shop5'),
        ('Hallway', 'Lounge1'),
        ('Lounge1', 'Lounge2'),
        ('Lounge2', 'MiniShop1'),
        ('MiniShop1', 'MiniShop2')
    ]
    G.add_edges_from(connections)
    return G


def create_transition_matrix(G):
    nodes = list(G.nodes)
    n = len(nodes)
    transition_matrix = np.zeros((n, n))
    for i, node in enumerate(nodes):
        neighbors = list(G.neighbors(node))
        if neighbors:
            prob = 1 / len(neighbors)
            for neighbor in neighbors:
                j = nodes.index(neighbor)
                transition_matrix[i][j] = prob
    return transition_matrix, nodes


# ----------------------------
# Случайное блуждание
# ----------------------------
def random_walk_with_transition_matrix(transition_matrix, nodes, start_node, steps, end_node=None):
    path = [start_node]
    current_node = start_node
    edge_visit_count = {}
    node_visit_count = defaultdict(int)

    current_index = nodes.index(start_node)
    node_visit_count[start_node] += 1

    for step in range(steps):
        probabilities = transition_matrix[current_index]
        if probabilities.sum() == 0:
            break  # Нет переходов

        next_index = np.random.choice(len(nodes), p=probabilities)
        next_node = nodes[next_index]
        path.append(next_node)

        edge = f"{current_node}-{next_node}"
        edge_visit_count[edge] = edge_visit_count.get(edge, 0) + 1
        node_visit_count[next_node] += 1

        if end_node and next_node == end_node:
            break

        current_node = next_node
        current_index = next_index

    return path, edge_visit_count, node_visit_count


# ----------------------------
# Координаты для визуализации
# ----------------------------
node_positions = {
    'Entrance': (1, 3),
    'Hallway': (3, 3),
    'Shop1': (3, 5),
    'Shop2': (5, 5),
    'Shop3': (7, 5),
    'Shop4': (9, 5),
    'Shop5': (11, 5),
    'Shop6': (13, 5),
    'FoodCourt': (3, 1),
    'Toilets': (5, 1),
    'Elevator': (3, 7),
    'Stairs': (5, 7),
    'Parking': (7, 7),
    'Office1': (9, 7),
    'Office2': (11, 7),
    'Exit': (15, 5),
    'Kiosk1': (7, 1),
    'Kiosk2': (8.5, 1),
    'RestArea1': (10, 1),
    'RestArea2': (11.5, 1),
    'InfoDesk': (11.5, 3),
    'Lounge1': (5, 3),
    'Lounge2': (6.5, 3),
    'MiniShop1': (8, 3),
    'MiniShop2': (9.5, 3)
}

floor_boundary = [(-1, -1), (17, -1), (17, 9), (-1, 9), (-1, -1)]


# ----------------------------
# Визуализация плана с путём и тепловой картой
# ----------------------------
def visualize_floorplan(G, path, edge_visit_count, node_visit_count, output_file='static/floorplan.png'):
    fig, ax = plt.subplots(figsize=(14, 8))
    boundary_x, boundary_y = zip(*floor_boundary)
    ax.plot(boundary_x, boundary_y, color='black', linewidth=2)

    # Отрисовка комнат
    for node, (x, y) in node_positions.items():
        # heatmap
        visits = node_visit_count.get(node, 0)
        # Нормировка для цветовой карты (0-1)
        max_visits = max(node_visit_count.values()) if node_visit_count.values() else 1
        norm_visits = min(visits / max_visits, 1)
        color = plt.cm.Reds(norm_visits)

        ax.add_patch(plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, edgecolor='black', facecolor=color))
        ax.text(x, y, f"{node}\n({visits})", fontsize=9, ha='center', va='center')

    # Связи
    for edge in G.edges():
        x1, y1 = node_positions[edge[0]]
        x2, y2 = node_positions[edge[1]]
        ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.7, linewidth=1)

    # Подсветка маршрута
    path_edges = []
    for i in range(len(path) - 1):
        path_edges.append((path[i], path[i + 1]))

    for edge in path_edges:
        x1, y1 = node_positions[edge[0]]
        x2, y2 = node_positions[edge[1]]
        ax.plot([x1, x2], [y1, y2], color='blue', linewidth=2, alpha=0.8)

    # Кол-во посещений ребер
    for edge, count in edge_visit_count.items():
        if count > 0:
            nodes = edge.split('-')
            if len(nodes) == 2:
                x1, y1 = node_positions[nodes[0]]
                x2, y2 = node_positions[nodes[1]]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(mid_x, mid_y, str(count), fontsize=10, color='green',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.set_xlim(-2, 16)
    ax.set_ylim(-2, 8)
    ax.axis('off')
    plt.title("Floorplan with Path and Heatmap")

    plt.savefig(output_file, dpi=100)
    plt.close(fig)
    return output_file


# ----------------------------
# KDE-анализ плотности посещений
# ----------------------------
def kde_density_analysis(node_visit_count):
    coords = []
    counts = []
    for node, count in node_visit_count.items():
        if node in node_positions:
            coords.append(node_positions[node])
            counts.append(count)

    if not coords:
        return {}

    coords = np.array(coords)
    counts = np.array(counts)

    # KDE с весами посещения
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(coords, sample_weight=counts)

    # Рассчёты плотности для каждой точки
    log_dens = kde.score_samples(coords)
    dens = np.exp(log_dens)

    # Нормализация плотности (0-1)
    max_dens = max(dens) if len(dens) > 0 else 1
    norm_dens = dens / max_dens

    # Сопоставляем с узлами
    dens_map = {}
    for i, node in enumerate(node_visit_count.keys()):
        if i < len(norm_dens):
            dens_map[node] = norm_dens[i]

    return dens_map


# ----------------------------
# Кластеризация паттернов
# ----------------------------
def cluster_patterns(path, nodes):
    # Преобразование пути в координаты
    coords = []
    for node in path:
        if node in node_positions:
            coords.append(node_positions[node])

    if len(coords) < 3:
        return {'kmeans': [], 'dbscan': []}

    coords = np.array(coords)

    # KMeans
    kmeans = KMeans(n_clusters=min(3, len(coords)), random_state=42).fit(coords)
    labels_kmeans = kmeans.labels_.tolist()

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=2).fit(coords)
    labels_dbscan = dbscan.labels_.tolist()

    return {
        'kmeans': labels_kmeans,
        'dbscan': labels_dbscan
    }


# ----------------------------
# Обнаружение аномалий
# ----------------------------
def detect_anomalies(path):
    if len(path) < 3:
        # Условие пути для анализа
        return {
            'one_class_svm': [1] * len(path),
            'isolation_forest': [1] * len(path),
            'gaussian': [1] * len(path)
        }

    # Преобразование пути в координаты и дополнительные признаки
    features = []
    for i, node in enumerate(path):
        if node in node_positions:
            x, y = node_positions[node]

            # 1. Расстояние от начальной точки
            start_x, start_y = node_positions[path[0]]
            dist_from_start = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)

            # 2. Количество предыдущих посещений этого узла
            prev_visits = path[:i].count(node)

            # 3. Количество уникальных посещенных узлов
            unique_visited = len(set(path[:i]))

            features.append([x, y, dist_from_start, prev_visits, unique_visited])

    if len(features) < 3:
        return {
            'one_class_svm': [1] * len(path),
            'isolation_forest': [1] * len(path),
            'gaussian': [1] * len(path)
        }

    X = np.array(features)

    # Стандартизация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Использование ансамбля подходов
    results = {
        'one_class_svm': [],
        'isolation_forest': [],
        'gaussian': []
    }

    # One-Class SVM с оптимизированными параметрами
    oc_svm = OneClassSVM(nu=0.05, gamma=0.1)
    svm_preds = oc_svm.fit_predict(X_scaled)
    results['one_class_svm'] = [1 if p == 1 else -1 for p in svm_preds]

    # Isolation Forest с контролем контаминации
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    if_preds = iso_forest.fit_predict(X_scaled)
    results['isolation_forest'] = [1 if p == 1 else -1 for p in if_preds]

    # Гауссовский подход
    try:
        gaussian = EllipticEnvelope(contamination=0.05)
        gaussian.fit(X_scaled)
        gaussian_preds = gaussian.predict(X_scaled)
        results['gaussian'] = [1 if p == 1 else -1 for p in gaussian_preds]
    except:
        results['gaussian'] = [1] * len(X_scaled)

    return results


# ----------------------------
# Анализ переходов и времени пребывания
# ----------------------------
def analyze_metrics(path, edge_visit_count, node_visit_count, step_time_sec=5):
    # Рассчёты уникальных посещений
    unique_visits = {}
    for node in set(path):
        unique_visits[node] = path.count(node)

    # Плотность посещений
    dens_map = kde_density_analysis(node_visit_count)

    # Форматирование таблицы
    metrics = []
    for node in unique_visits:
        metrics.append({
            'zone': node,
            'visit_count': unique_visits[node],
            'avg_time_sec': unique_visits[node] * step_time_sec,
            'density': dens_map.get(node, 0)
        })

    metrics.sort(key=lambda x: x['visit_count'], reverse=True)

    return metrics


# ----------------------------
# Инициализация
# ----------------------------
graph = create_graph()
transition_matrix, nodes = create_transition_matrix(graph)


# ----------------------------
# Flask routes
# ----------------------------
@app.route('/')
def index():
    return send_file('index.html')


@app.route('/random-walk', methods=['POST'])
def random_walk():
    data = request.json
    start_room = data.get('start_room')
    end_room = data.get('end_room')
    steps = data.get('steps', 10)

    if start_room not in graph.nodes:
        return jsonify({"error": "Invalid start room"}), 400
    if end_room and end_room not in graph.nodes:
        return jsonify({"error": "Invalid end room"}), 400

    path, edge_visit_count, node_visit_count = random_walk_with_transition_matrix(
        transition_matrix, nodes, start_room, steps, end_room
    )

    # Генерирация уникальный хеш путей для кеширования
    path_hash = hashlib.md5(''.join(path).encode()).hexdigest()
    output_file = f"static/floorplan_{path_hash}.png"

    if not os.path.exists('static'):
        os.makedirs('static')

    # Генерирация изображения
    if not os.path.exists(output_file):
        visualize_floorplan(graph, path, edge_visit_count, node_visit_count, output_file)

    # Анализ плотности KDE
    kde_density = kde_density_analysis(node_visit_count)

    # Кластеризация
    clustering = cluster_patterns(path, nodes)

    # Аномалии
    anomalies = detect_anomalies(path)

    # Метрики
    metrics = analyze_metrics(path, edge_visit_count, node_visit_count)


    return jsonify({
        "visualization_url": "/" + output_file,
        "edge_visit_counts": edge_visit_count,
        "node_visit_count": node_visit_count,
        "path": path,
        "kde_density": kde_density,
        "clustering": clustering,
        "anomalies": anomalies,
        "metrics": metrics
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)