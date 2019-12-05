import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import matplotlib.pyplot as plt
import itertools

from queue import Queue
from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

class Solver:

    def __init__(self, list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
        self.list_of_locations = list_of_locations
        self.list_of_homes = list_of_homes
        self.starting_car_location = starting_car_location
        self.adjacency_matrix = adjacency_matrix
        self.params = params
        self.graph, _ = adjacency_matrix_to_graph(adjacency_matrix)

        self.dijkstraData = list(nx.all_pairs_dijkstra_path_length(self.graph))
        self.threshDict = self.create_threshold_dictionary()
        self.groupDict = {}



    # ONLY PASS IN GRAPH VERTEX INDICIES


    def greedy_set_cover(self):
        toDropOff = [self.get_graph_node(elem) for elem in self.list_of_homes]
        dropOffDict = {}
        path = []

        vertexSetWeights = [0 for _ in range(len(self.list_of_locations))] # People/Energy ratio
        vertexSetGroups = [0 for _ in range(len(self.list_of_locations))]


        for vertexData in self.groupDict:

            for setThing in self.groupDict[vertexData]:
                # print(vertexData) # Vertex to drop off at
                # print(setThing) # People to drop off
                # print(self.groupDict[vertexData][setThing]) # Weight for these people to walk home
                currVertex = vertexData
                dropoffs = setThing
                weight = self.groupDict[vertexData][setThing]
                if weight != 0:
                    vertexSetWeights[currVertex] = len(dropoffs) / weight
                    vertexSetGroups[currVertex] = dropoffs

                # Get what has the most to drop off, and the max person/energy ratio

        maxVal = max(vertexSetWeights) # Getting max ratio valued vertex 
        maxVertex = vertexSetWeights.index(maxVal) # Vertex associated with this ratio

        dropOffDict[maxVertex] = vertexSetGroups[maxVertex] # Store who is dropped off where

        for elem in vertexSetGroups[maxVertex]: # Update toDropOff
            toDropOff.remove(elem)

        # Assign most optimal tuple to dropoff vertex
        # Get weight of max vertex
        maxVertexWeight = self.groupDict[maxVertex][vertexSetGroups[maxVertex]]
        
        # Update vertexSetWeights
        while len(toDropOff) != 0:

            # UPDATE VERTEX SET WEIGHTS     
            for i in range(len(vertexSetWeights)):
                if vertexSetGroups[i] != 0:
                    vertexSetGroups[i] = tuple(x for x in vertexSetGroups[i] if x in toDropOff)
                    if self.get_dist_tuple(i, vertexSetGroups[i]) == 0 and len(vertexSetGroups[i]) == 1:
                        vertexSetWeights[i] = 1
                    elif self.get_dist_tuple(i, vertexSetGroups[i]) == 0:
                        vertexSetWeights[i] = 0
                    else:

                        # print(vertexSetWeights[i])
                        vertexSetWeights[i] = len(vertexSetGroups[i]) / self.get_dist_tuple(i, vertexSetGroups[i])
                        # print(vertexSetWeights[i])


            # print(vertexSetGroups)
            # print(vertexSetWeights)
            # print(toDropOff)

            maxVal = max(vertexSetWeights) # Getting max ratio valued vertex 
            maxVertex = vertexSetWeights.index(maxVal) # Vertex associated with this ratio

            dropOffDict[maxVertex] = vertexSetGroups[maxVertex] # Store who is dropped off where

            if vertexSetGroups[maxVertex] != 0:
                for elem in vertexSetGroups[maxVertex]: # Update toDropOff
                    toDropOff.remove(elem)

        # print(toDropOff)
        # WHILE LOOP FOR PATH
        # print(dropOffDict)
        # Find next optimal set based on ratio. Save this vertex, update our vertexSetWeights



        return dropOffDict


    def nearest_neighbor_paths(self, dropOffDict):
        paths = [[x] for x in list(dropOffDict.keys())]

        for path in paths:
            nodes = list(dropOffDict.keys())
            nodes.remove(path[0])
            while len(nodes):
                nodeCopy = nodes
                temp = [self.get_dist(path[len(path) - 1], x) for x in nodes]
                minVal = min(temp)
                minIndex = temp.index(minVal)

                path.append(nodes[minIndex])
                nodes.pop(minIndex)


        costs = []
        for path in paths:
            cost = 0
            for i in range(len(path) - 1):
                cost += self.get_dist(i, i + 1)

            cost += self.get_dist(path[0], self.get_graph_node(self.starting_car_location)) + self.get_dist(path[len(path) - 1], self.get_graph_node(self.starting_car_location))
            costs.append(cost)

        # print(min(costs))
        # print(costs.index(min(costs)))

        print(paths)
        print(costs)
        return paths[costs.index(min(costs))], dropOffDict



    def enumerate_paths(self, dropOffDict):
        paths = []
        costs = []
        dropoffs = dropOffDict.keys()

        permutations = list(itertools.permutations(dropoffs))

        # print(len(permutations))
        for path in permutations:
            # print('working')
            currPath = [self.starting_car_location] + [self.list_of_locations[i] for i in path] + [self.starting_car_location]
            cost = 0
            for i in range(len(path) - 1):
                cost += self.get_dist(i, i + 1)

            cost += (self.get_dist(self.get_graph_node(self.starting_car_location), path[0])) + (self.get_dist(path[len(path) - 1], self.get_graph_node(self.starting_car_location)))
            paths.append(path)
            costs.append(cost)

        index = cost.index(min(cost))
        # print(paths[index])


    def get_dist_tuple(self, start, end_nodes):

        retVal = 0
        for elem in end_nodes:
            retVal += self.dijkstraData[start][1][elem]
        return retVal


    def group_nodes(self, vertex):

        currGroup = []
        self.groupDict[vertex] = {}
        for home in self.list_of_homes:
            nextHome = self.get_graph_node(home)
            if self.soda_dist_comparator(vertex, nextHome):
                currGroup.append(nextHome)
                #If this returns true, dist from vertex to neighbor < dist soda_neighbor
            
        totalCost = sum([self.get_dist(vertex, i) for i in currGroup])
        self.groupDict[vertex][tuple(currGroup)] = totalCost


    def create_threshold_dictionary(self):
    
        retDict = {}
        for i in range(len(self.dijkstraData)):
            retDict[i] = self.dijkstraData[i][1][self.get_graph_node(self.starting_car_location)]

        return retDict


    def get_dist(self, start, end):
        indexStart = start
        indexEnd = end
        return self.dijkstraData[indexStart][1][indexEnd]


    def get_graph_node(self, name):
        # Returns the graph node number for vertex of name "name"
        return self.list_of_locations.index(name)


    def soda_dist_comparator(self, start, end):
        # Returns if dist from start to end < dist from car_start_location to end

        thresh = self.threshDict[end]
        start_end_dist = self.get_dist(start, end)
        return start_end_dist < thresh


def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """

    solver = Solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params)

    for location in list_of_locations:
        solver.group_nodes(solver.get_graph_node(location))

    path, dropOffs = solver.nearest_neighbor_paths(solver.greedy_set_cover())

    path = [solver.get_graph_node(solver.starting_car_location)] + path + [solver.get_graph_node(solver.starting_car_location)]

    print(path)


    # DROPOFF PATH VALID
    newPath = []
    for i in range(len(path) - 1):
        newPath += nx.dijkstra_path(solver.graph, path[i], path[i + 1])[:-1]

    newPath.append(newPath[0])
    print(newPath)

    print(cost_of_solution(solver.graph, newPath, dropOffs))



    return newPath, dropOffs

    # print(solver.list_of_locations)
    # print(solver.list_of_homes)
    # print(solver.starting_car_location)
    # print(solver.adjacency_matrix)
    # print(solver.params)
    # print(solver.graph)
    # print(solver.dijkstraData)
    # print(solver.threshDict)
    ## SODA HALL IS ALWAYS NODE 0



    pass






def get_graph_node(name, list_of_locations):
    # Returns the graph node number for vertex of name "name"
    return list_of_locations.index(name)

def create_threshold_dictionary(start, dijkstraData, list_of_locations):
    
    retDict = {}
    for i in range(len(dijkstraData)):
        retDict[i] = dijkstraData[i][1][get_graph_node(start, list_of_locations)]

    return retDict
    # Should be a dictionary where index i is dist from Soda to graph location i




def group_homes_at_vertices(graph, list_of_homes, thresh_dict):

    for node in graph.nodes:
        return False


def group_homes_at_node(node, list_of_homes, thresh_dict, graph):
    print("cat")

def bfs_with_limit(start, list_of_targets, threshold, graph, adjacency_matrix):
    
    # ENSURE THAT LIST OF TARGETS ALIGNS

    marked = [0 for _ in range(len(graph.nodes))]
    dist = [0 for _ in range(len(graph.nodes))]
    fifo = Queue()
    edges = nx.edges(graph)

    fifo.put(start)
    marked[start] = 1
    while not fifo.empty():
        currNode = fifo.get()
        for neighbor in nx.all_neighbors(graph, currNode):
            if marked[neighbor] == 0 and (dist[currNode] + adjacency_matrix[currNode][neighbor] <= threshold[neighbor]):
                fifo.put(neighbor)
                marked[neighbor] = 1
                dist[neighbor] = dist[currNode] + adjacency_matrix[currNode][neighbor]



def find_mst_end_nodes(mst):
    end_nodes = []
    for node in mst.degree:
        if node[1] == 1:
            end_nodes.append(node[0])

    closest_to_end_nodes = []
    for edge in mst.edges:
        if edge[0] in end_nodes and edge[1] not in closest_to_end_nodes:
            closest_to_end_nodes.append((edge[0], edge[1]))
        elif edge[1] in end_nodes and edge[0] not in closest_to_end_nodes:
            closest_to_end_nodes.append((edge[1], edge[0]))

    return closest_to_end_nodes




def get_shortest_paths(graph):
    
    # Lengths is a generator of tuples. Tuple[0] = node name, Tuple[1] = dictionary of other nodes and associated distances
    lengths = nx.all_pairs_bellman_ford_path_length(graph)
    


    most_used_dict = {}
    closest_dists = []

    for node in nx.nodes(graph):
        most_used_dict[node] = 0

    for item in lengths:
        minimum = 10000
        vert = 0

        for key in item[1].keys():
            if item[1][key] < minimum and item[1][key] != 0:
                minimum = item[1][key]
                vert = key

        closest_dists.append(vert)

    print(closest_dists)

    for elem in closest_dists:
        most_used_dict[elem] = most_used_dict[elem] + 1

    print(most_used_dict)


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
