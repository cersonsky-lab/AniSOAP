import numpy as np

class TrivariateMonomialIndices:
    """
    Class for generating an iterator object over all trivariate 
    monomials of the form f(x,y,z) = x^n0 * y^n1 * z^n2
    sorted in the lexicographical order.
    
    Without this class, iterating over all monomials at some fixed degree
    requires the use of a double loop of the form:
    
    idx = 0
    for n0 in range(deg, -1, -1):
            for n1 in range(deg-n0, -1, -1):
            n2 = deg - n0 - n1
            
            ... # do something with exponents (n0,n1,n2)
            
            idx += 1
    
    Instead, with this class, these lines can be reduced to
    myiter = iter(TrivariateMonomialIndices(deg=2))
    for idx, n0, n1, n2 in myiter:
        ... # do something with exponents (n0, n1, n2)
        
    """
    def __init__(self, deg):
        self.num_indices = (deg+1)*(deg+2)//2
        
        self.exponent_list = []
        idx = 0
        for n0 in range(deg, -1, -1):
            for n1 in range(deg-n0, -1, -1):
                n2 = deg - n0 - n1
                self.exponent_list.append((n0,n1,n2))
                idx += 1
        
    def __iter__(self):
        self.idx_flattened = 0        
        return self

    def __next__(self):
        current_idx = self.idx_flattened
        
        if current_idx < self.num_indices:
            self.idx_flattened += 1            
            n0, n1, n2 = self.exponent_list[current_idx]
            return (current_idx, n0, n1, n2)
        else:
            raise StopIteration
    
    def find_idx(self, exponents):
        """
        

        Parameters
        ----------
        exponents : 3-tuple (n0, n1, n2)
            The exponents of the monomial x^n0 * y^n1 * z^n2

        Returns
        -------
        The index of the tuple in the lexicographical order

        """
        assert n0 + n1 + n2 == self.deg
        return self.exponent_list.index(exponents)
    
    def get_exponents(self, idx):
        return self.exponent_list[idx]


if __name__ == '__main__':
	# Example for how to use the iterator
	monomial_iterator = TrivariateMonomialIndices(deg = 2)
	myiter = iter(monomial_iterator)

	for idx, n0, n1, n2 in myiter:
		print(f'idx = {idx}, (n0,n1,n2) = {n0,n1,n2}')
