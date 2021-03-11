import numpy as np
np.set_printoptions(threshold=np.inf)


class GeneticAlgorithm:
    def __init__(self, func, n_dim, n_select, size_pop=100,
                 max_iter=200, prob_mutation=0.001, seed=0):
        """0-1 genetic algorithm with weight constraint for minimization.
        The implementation is referenced in:
            https://codechina.csdn.net/mirrors/guofei9987/scikit-opt
        Args:
            func: function
                The function to optimize.
            n_dim: int
                Number of variables of function.
            n_select: int
                Number of variables set to 1.
            size_pop: int
                Size of population.
            max_iter: int
                Maximum number of iterations.
            prob_mutation : float between 0 and 1
                Probability of mutation.
            seed: int
                Seed for random generator.
        Attributes:
            Lind: array_like
                 The number of genes of every variable of function (segments).
            generation_best_X : array_like. Size is max_iter.
                Best X of every generation.
            generation_best_ranking : array_like. Size if max_iter.
                Best ranking of every generation.
        """
        self.func = func
        self.n_dim = n_dim
        self.n_select = n_select
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.prob_mutation = prob_mutation
        self.seed = seed
        np.random.seed(self.seed)

        self.chromosome = None        # Shape = (size_pop, n_dim)
        self.func_output = None       # Shape = (size_pop,)
        self.fit_vals = None          # Shape = (size_pop,)

        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_fitvals = []

        self.best_x, self.best_y = None, None

        self.chromosome = self.create_population()

    def create_population(self):
        """"Create the population."""
        chromosome_ = np.zeros((self.size_pop, self.n_dim))

        for p in range(self.size_pop):
            mask_ = np.random.choice(
                range(self.n_dim), self.n_select, replace=False)
            chromosome_[p][mask_] = 1

        return chromosome_.astype(int)

    def ranking(self, func_output):
        # """Linear ranking selection."""
        # return np.argsort(np.argsort(-func_output))
        return -func_output

    def selection(self, tournament_size=3):
        """Select the best individuals by tournament selection.
        Note that the population size is reduced after eliminate,
        so we increase the population size here to keep it unchanged.
        """
        size_pop_, _ = self.chromosome.shape
        aspirants_idx = np.random.randint(
            size_pop_, size=(self.size_pop, tournament_size))
        aspirants_values = self.fit_vals[aspirants_idx]
        winner = aspirants_values.argmax(axis=1)
        select_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
        self.chromosome = self.chromosome[select_index, :]

    def crossover(self):
        """Crossover at the points n1 to n2."""
        for i in range(0, self.size_pop, 2):
            diff_gene_idx, = np.where(
                self.chromosome[i, :] != self.chromosome[i+1, :])

            if len(diff_gene_idx) == 0:
                continue

            n1, n2 = np.sort(
                np.random.choice(diff_gene_idx, 2, replace=False))

            seg1 = self.chromosome[i, n1:(n2+1)].copy()
            seg2 = self.chromosome[i+1, n1:(n2+1)].copy()

            self.chromosome[i, n1:(n2+1)] = seg2
            self.chromosome[i+1, n1:(n2+1)] = seg1

    def mutation(self):
        """Weight conservation mutation."""
        mask = (np.random.rand(self.size_pop) < self.prob_mutation)
        chroms_to_mutate = self.chromosome[mask]

        for chrom in chroms_to_mutate:
            num_dominant_genes = chrom.sum()

            # mutate 0 to 1
            if num_dominant_genes < self.n_select:
                num_mutate = self.n_select - num_dominant_genes
                possible_mutate_idx, = np.where(chrom == 0)
                mutate_idx = np.random.choice(
                    possible_mutate_idx, num_mutate, replace=False)
                chrom[mutate_idx] = 1

            # mutate 1 to 0
            elif num_dominant_genes > self.n_select:
                num_mutate = num_dominant_genes - self.n_select
                possible_mutate_idx, = np.where(chrom == 1)
                mutate_idx = np.random.choice(
                    possible_mutate_idx, num_mutate, replace=False)
                chrom[mutate_idx] = 0

            # dual mutate
            else:
                num_mutate = 1

                # mutate 0 to 1
                possible_mutate_idx, = np.where(chrom == 0)
                mutate_idx = np.random.choice(
                    possible_mutate_idx, num_mutate, replace=False)
                chrom[mutate_idx] = 1

                # mutate 1 to 0
                possible_mutate_idx, = np.where(chrom == 1)
                mutate_idx = np.random.choice(
                    possible_mutate_idx, num_mutate, replace=False)
                chrom[mutate_idx] = 0

    def eliminate(self):
        """Eliminate populations that violate weight constraint,
        i.e., the number of 1 should be n_select.
        """
        weight_mask = (self.chromosome.sum(axis=1) == self.n_select)
        self.chromosome = self.chromosome[weight_mask]

    def fit(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter

        for i in range(self.max_iter):
            # Calculate fit values
            self.func_output = self.func(self.chromosome)
            self.fit_vals = self.ranking(self.func_output)

            # Record the best ones
            generation_best_index = self.fit_vals.argmax()
            self.generation_best_X.append(
                self.chromosome[generation_best_index, :])
            self.generation_best_Y.append(
                self.func_output[generation_best_index])
            self.all_history_Y.append(self.func_output)
            self.all_history_fitvals.append(self.fit_vals)

            # Selection, crossover, and mutation
            self.selection()
            self.crossover()
            self.mutation()

            # Eliminate invalid ones
            self.eliminate()

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]))
        return self.best_x, self.best_y


if __name__ == "__main__":
    """An example"""
    ga = GeneticAlgorithm(
        func=lambda x: x.cumsum(axis=1).sum(axis=1),
        n_dim=35,
        n_select=8,
        size_pop=100,
        max_iter=1000,
        prob_mutation=0.001,
        seed=0
    )
    print(ga.fit())
