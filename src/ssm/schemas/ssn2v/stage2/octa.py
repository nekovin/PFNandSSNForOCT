def compute_octa(oct1, oct2):

    numerator = (oct1 - oct2)**2
    denominator = oct1**2 + oct2**2
    
    epsilon = 1e-10
    octa = numerator / (denominator + epsilon)
    
    return octa