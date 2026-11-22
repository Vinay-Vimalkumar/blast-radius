def distribute(numbers, millis):
    ordered = sorted(numbers, key=lambda n: (-millis[n], n))
    if not ordered:
        return []
    total = sum(millis[n] for n in ordered)
    largest = millis[ordered[0]]
    best_cost = total + 2000
    best = [ordered]
    for count in range(2, min(len(ordered), 8) + 1):
        lower_bound = max(largest, (total + count - 1) // count)
        if lower_bound + 2000 * count >= best_cost:
            continue
        workers = [[] for _ in range(count)]
        loads = [0] * count
        for number in ordered:
            worker = loads.index(min(loads))
            workers[worker].append(number)
            loads[worker] += millis[number]
        cost = max(loads) + 2000 * count
        if cost < best_cost:
            best, best_cost = workers, cost
    return best


class SpareCapacity:
    def __init__(self, millis):
        self.millis = millis
        self.ordered = sorted(millis, key=lambda item: (millis[item], item))
        self.full = distribute(self.ordered, millis)
        self.full_cost = self.cost(self.full)

    def cost(self, workers):
        loads = (sum(self.millis[n] for n in group) for group in workers)
        return 2000 * len(workers) + max(loads, default=0)

    def fill(self, workers):
        if not workers:
            return workers
        loads = [sum(self.millis[n] for n in group) for group in workers]
        ceiling = max(loads)
        if self.full_cost <= ceiling + 2000 * len(workers):
            return self.full
        filled = [group.copy() for group in workers]
        chosen = {n for group in filled for n in group}
        room = [ceiling - load for load in loads]
        for number in self.ordered:
            duration = self.millis[number]
            if duration > max(room):
                break
            if number in chosen:
                continue
            target = min((i for i, capacity in enumerate(room) if capacity >= duration),
                         key=room.__getitem__)
            filled[target].append(number)
            room[target] -= duration
        return filled
