import heapq

def update_nodes_pipes_count(topology_in: dict):
    '''
    Функция пересчитывает количество объектов pipe и узлов, обновляя их счетчик в топологии
    '''
    topology_in['globalNodesCount'] = len(topology_in['nodes'])
    topology_in['globalPipesCount'] = len(topology_in['pipes'])

def Dijkstra_algorithm(topology_in: dict, start_node: int, end_node: int):
    '''
    Функция реализует алгоритм Дейкстры. 

    :param topology: топология
    :param start_node: стартовый узел, из которого начинаем путь
    :param end_node: конечный узел, до которого ищем путь

    :return path: словарь с id узлов, которые входят в наикратчайший путь
    :return used_elements: словарь с id pipe, которые входят в наикратчайший путь
    '''
    graph = {}

    # Формируем словарь графа для топологии по типу: 
    # {узел_1: [(узел_2, элемент_соединяющий_узел_1_узел_2), (), ...], узел_2: [(), (), ...], ...}
    for pipe in topology_in['pipes']:
        graph.setdefault(pipe['inletNodeId'], []).append((pipe['outletNodeId'], pipe['id']))
        graph.setdefault(pipe['outletNodeId'], []).append((pipe['inletNodeId'], pipe['id']))
    
    print(f'graph: {graph}')

    # Очередь с приоритетом, содержащая кортежи:
    # (текущее расстояние, текущий узел, путь, использованные элементы)
    queue = [(0, start_node, [start_node], [])]

    # Словарь уже посещенных узлов сети
    visited_nodes = {}

    # Цикл по всем узлам в очереди
    while queue:
        # Извлекаем узел с наименьшим расстоянием
        dist, node, path, used_elements = heapq.heappop(queue)

        if node in visited_nodes and visited_nodes[node] <= dist:
            continue
        
        # Обновляем минимальное расстояние до текущего узла
        visited_nodes[node] = dist

        # Если текущий узел является последним, то перываем цикл
        if node == end_node:
            return path, used_elements
        
        # Обходим всех соседей текущего узла
        for neighbor_node, element in graph.get(node, []):
            if neighbor_node not in visited_nodes:
                # Добавляем новых соседей в очередь
                heapq.heappush(queue, (dist + 1, neighbor_node, path + [neighbor_node], used_elements + [element]))

    return None # Если путь не найден 

def get_unique_ids(topology_in_in: dict) -> set:
    '''
    Функция возвращает множества уникальных номеров id для узлов и объектов pipes топологии

    :param topology_in_in: топология сети
    '''
    unique_nodes_ids = set()
    for i in topology_in_in['nodes']:
        unique_nodes_ids.add(i['id'])

    unique_pipes_ids = set()
    for i in topology_in_in['pipes']:
        unique_pipes_ids.add(i['id'])

    return unique_nodes_ids, unique_pipes_ids

def update_unique_nodes_ids(unique_nodes_ids: set, nodes_ids: list) -> set:
    '''
    Функция добавляет в множество новые уникальные значения id
    '''
    unique_nodes_ids.update(nodes_ids)
    return unique_nodes_ids

def update_unique_pipes_ids(unique_pipes_ids: set, pipes_ids: list) -> set:
    unique_pipes_ids.update(pipes_ids)
    return unique_pipes_ids

def form_pipes_based_on_section(topology_in: dict, tube_sections_in: dict):
    '''
    Функция обновляет топологию, создавая объекты труб и новые мнимые узлы на основе входного файла с разбиением трубы по секциям
        !!!Функция при это также обновляет топологию, которая была подана на вход. 
        Если нужно сохранить изначальную топологию, то нужно подавать на вход ее глубокую копию (copy,deepcopy())!!!
    :param topology_in: топология сети
    :param tube_sections_in: файл с разбиением по секциям

    :param retur: обновленная топология
    '''

    # Количество секций трубы
    number_of_sections = len(tube_sections_in['sections'])

    # Входной и выходной узлы трубы, разбитой на секции
    inletNodeId = tube_sections_in['inletNodeId']
    outletNodeId = tube_sections_in['outletNodeId']
    
    # Получаем множества уникальных значений id узлов и объектов pipe топологии
    unique_nodes_ids, unique_pipes_ids = get_unique_ids(topology_in)

    # Проверка на существование входного и вызодного узлов в топологии
    if inletNodeId not in unique_nodes_ids:
        print(f'Узел с номером id {inletNodeId} не существует в топологии')
        return
    elif outletNodeId not in unique_nodes_ids:
        print(f'Узел с номером id {outletNodeId} не существует в топологии')
        return
    
    # Создаем список всех объектов pipe и узлов через копии топологии, чтобы итерироваться по ним в цикле 
    topology_in_pipes = topology_in['pipes'].copy() 
    topology_in_nodes = topology_in['nodes'].copy()

    # Формируем список промежуточных узлов и элементов, которые соединяют входной и выходной узлы входящего файла, если такие есть
    intermediate_nodes_ids, intermediate_pipes_ids = Dijkstra_algorithm(topology_in, inletNodeId, outletNodeId)

    print(f'intermediate_nodes_ids : {intermediate_nodes_ids}')
    print(f'intermediate_pipes_ids: {intermediate_pipes_ids}')

    intermediate_nodes_ids = intermediate_nodes_ids[1:-1]

    for node in topology_in_nodes:
        if node['id'] in intermediate_nodes_ids:
            print('yes')
            if 'Imaginery' in node['name']:
                for node_id in intermediate_nodes_ids:
                    for node_1 in topology_in_nodes:
                        if node_1['id'] == node_id:
                            topology_in['nodes'].remove(node_1)
                for pipe_id in intermediate_pipes_ids:
                    for pipe in topology_in_pipes:
                        if pipe['id'] == pipe_id:
                            topology_in['pipes'].remove(pipe)                
                break
    
    '''
    Цикл проверяет, нет ли объекта трубы в изначальной топологии, который соединяет указанные во входном файле с секциями узлы.
    Если есть - удаляет этот объект.
    '''
    for pipe in topology_in_pipes:
        inletNodeId_pipe = pipe['inletNodeId']
        outletNodeId_pipe = pipe['outletNodeId']
        if inletNodeId_pipe == inletNodeId and outletNodeId_pipe == outletNodeId:
            # Удаляем обхект pipe
            topology_in['pipes'].remove(pipe)

    # Получаем множества уникальных значений id узлов и объектов pipe топологии
    unique_nodes_ids, unique_pipes_ids = get_unique_ids(topology_in)

    # Цикл по всем секциям
    for i in range(number_of_sections):
        # Текущая секция 
        section = tube_sections_in['sections'][i]
        
        
        if number_of_sections == 1:     # Проверка на то, одна ли секция во входном файле
            inletNodeId_i = inletNodeId
            outletNodeId_i = outletNodeId
        else:
            # Номера id входных и выходных узлов новой трубы, основанной на текущей секции
            if i == 0:
                inletNodeId_i = inletNodeId
                outletNodeId_i = max(unique_nodes_ids) + 1
            elif i == number_of_sections - 1:
                outletNodeId_i = outletNodeId
                inletNodeId_i = topology_in['pipes'][-1]['outletNodeId']
            else:            
                inletNodeId_i = topology_in['pipes'][-1]['outletNodeId']
                outletNodeId_i = max(unique_nodes_ids) + 1

        # Создаем новый промежуточный узел
        if outletNodeId_i not in unique_nodes_ids:
            new_node = {}
            new_node['id'] = outletNodeId_i
            new_node['name'] = f"Imaginery_Junction_{outletNodeId_i}"
            new_node['type'] = "JUNCTION"

            topology_in['nodes'].append(new_node)

        # Добавляем значения id новых мнимых усзлов в множество уникальных значений id узлов топологии
        unique_nodes_ids = update_unique_nodes_ids(unique_nodes_ids, [inletNodeId_i, outletNodeId_i])

        # Номер id новой трубы, основанной на текущей секции
        if len(unique_pipes_ids) != 0:
            new_pipe_id = max(unique_pipes_ids) + 1
        else:
            new_pipe_id = 0

        # Добавляем значениt id новой трубы (текущей секции) в множество уникальных значений id объектов pipes топологии
        unique_pipes_ids = update_unique_pipes_ids(unique_pipes_ids, [new_pipe_id]) 

        # Считываем параметры секции
        l = section['length_m'] # Длина секции 
        d_h = section['height_difference_m'] # Перепад выост секции
        diameter = section['innerDiameterMm'] # Внутренний диаметр новой трубы
        roughnessMm = section['roughnessMm'] # Шероховатость новой трубы

        '''
        Если это первая секция, то начальная координата [0, 0], вторая расчитывается через длину и перепад высоты.
        Если это не первая секция, то в качестве начальной координаты берется крайняя координата предыдущей секции, 
        вторая рассчитывается аналогично. Сами координаты не важны, в дальнейшем они используются для расчета наклона и длины трубы.
        '''
        if i == 0:
            x_0 = 0
            h_0 = 0
        else:
            # Берем профиль добавленной трубы на прошлой итерации и берем ее крайнюю координату
            profle_arr_of_previous_section = topology_in['pipes'][-1]['profileHorDistanceMSpaceHeightM']
            x_0_str, h_0_str = profle_arr_of_previous_section[1].split()
            x_0 = float(x_0_str)
            h_0 = float(h_0_str)

        x_1 = x_0 + (l**2 - d_h**2) ** 0.5
        h_1 = h_0 + d_h

        profileHorDistanceMSpaceHeightM = [f'{x_0} {h_0}', f'{x_1} {h_1}'] # Профиль 

        # Создаем и заполняем словарь объекта новой трубы
        new_pipe = {}
        
        new_pipe['id'] = new_pipe_id
        new_pipe['innerDiameterMm'] = diameter
        new_pipe['name'] = f'pipe_{inletNodeId}_{outletNodeId}_section_{i}'
        new_pipe['type'] = 'TUBE'
        new_pipe['inletNodeId'] = inletNodeId_i
        new_pipe['outletNodeId'] = outletNodeId_i
        new_pipe['profileHorDistanceMSpaceHeightM'] = profileHorDistanceMSpaceHeightM
        new_pipe['roughnessMm'] = roughnessMm

        # Добавляем трубу в топологию 
        topology_in['pipes'].append(new_pipe)

    update_nodes_pipes_count(topology_in)

    return topology_in