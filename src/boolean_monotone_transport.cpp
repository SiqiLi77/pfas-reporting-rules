#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <vector>

namespace {

struct Edge {
    int to;
    int reverse;
    double capacity;
};

void add_edge(std::vector<std::vector<Edge>>& graph, int from, int to, double capacity) {
    Edge forward{to, static_cast<int>(graph[to].size()), capacity};
    Edge reverse{from, static_cast<int>(graph[from].size()), 0.0};
    graph[from].push_back(forward);
    graph[to].push_back(reverse);
}

double max_flow_one(const double* lower, const double* upper) {
    constexpr int states = 64;
    constexpr int source = 128;
    constexpr int sink = 129;
    constexpr int nodes = 130;
    constexpr double epsilon = 1e-14;
    std::vector<std::vector<Edge>> graph(nodes);
    for (int y = 0; y < states; ++y) {
        add_edge(graph, source, y, std::max(0.0, lower[y]));
        for (int z = 0; z < states; ++z) {
            if ((z & y) == z) {
                add_edge(graph, y, states + z, 1.0);
            }
        }
    }
    for (int z = 0; z < states; ++z) {
        add_edge(graph, states + z, sink, std::max(0.0, upper[z]));
    }

    double total = 0.0;
    std::vector<int> level(nodes), cursor(nodes);
    while (true) {
        std::fill(level.begin(), level.end(), -1);
        std::queue<int> queue;
        level[source] = 0;
        queue.push(source);
        while (!queue.empty()) {
            int node = queue.front();
            queue.pop();
            for (const auto& edge : graph[node]) {
                if (edge.capacity > epsilon && level[edge.to] < 0) {
                    level[edge.to] = level[node] + 1;
                    queue.push(edge.to);
                }
            }
        }
        if (level[sink] < 0) break;
        std::fill(cursor.begin(), cursor.end(), 0);
        auto dfs = [&](auto&& self, int node, double flow) -> double {
            if (node == sink) return flow;
            for (int& index = cursor[node]; index < static_cast<int>(graph[node].size()); ++index) {
                Edge& edge = graph[node][index];
                if (edge.capacity <= epsilon || level[edge.to] != level[node] + 1) continue;
                double sent = self(self, edge.to, std::min(flow, edge.capacity));
                if (sent > epsilon) {
                    edge.capacity -= sent;
                    graph[edge.to][edge.reverse].capacity += sent;
                    return sent;
                }
            }
            return 0.0;
        };
        while (true) {
            double sent = dfs(dfs, source, 1.0 - total);
            if (sent <= epsilon) break;
            total += sent;
            if (total >= 1.0 - epsilon) return total;
        }
    }
    return total;
}

}  // namespace

extern "C" int boolean_monotone_batch(
    const double* lower,
    const double* upper,
    std::size_t observations,
    double tolerance,
    double* maximum_untransported,
    std::size_t* infeasible_count
) {
    if (!lower || !upper || !maximum_untransported || !infeasible_count) return 1;
    double maximum = 0.0;
    std::size_t failures = 0;
    for (std::size_t index = 0; index < observations; ++index) {
        double flow = max_flow_one(lower + index * 64, upper + index * 64);
        double residual = std::max(0.0, 1.0 - flow);
        maximum = std::max(maximum, residual);
        failures += static_cast<std::size_t>(residual > tolerance);
    }
    *maximum_untransported = maximum;
    *infeasible_count = failures;
    return 0;
}
