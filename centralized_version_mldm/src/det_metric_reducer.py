class MyAvgMetricReducer(pytorch.MetricReducer):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.counts = 0

    # User-defined mechanism for collecting values throughout
    # training or validation. This update() mechanism demonstrates
    # a computationally- and memory-efficient way to store the values.
    def update(self, value):
        self.sum += sum(value)
        self.counts += 1

    def per_slot_reduce(self):
        # Because the chosen update() mechanism is so
        # efficient, this is basically a noop.
        return self.sum, self.counts

    def cross_slot_reduce(self, per_slot_metrics):
        # per_slot_metrics is a list of (sum, counts) tuples
        # returned by the self.pre_slot_reduce() on each slot
        sums, counts = zip(*per_slot_metrics)
        return sum(sums) / sum(counts)


# class MyPyTorchTrial(pytorch.PyTorchTrial):
#     def __init__(self, context):
#         # Register your custom reducer.
#         self.my_avg = context.wrap_reducer(
#             MyAvgMetricReducer(), name="my_avg"
#         )
#         ...

#     def train_batch(self, batch, epoch_idx, batch_idx):
#         ...
#         # You decide how/when you call update().
#         self.my_avg.update(my_val)

#         # The "my_avg" metric will be included in the final
#         # metrics after the workload has completed; no need
#         # to return it here.
#         return {"loss": loss}
