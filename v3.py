@cuda.jit
def v3(factorial_number, alternatives, profile, stride, total):
  
  ############### GET THE FACTORIAL NUMBER FROM IDX ############################

  # get the number of elements
  n = factorial_number.shape[1]

  # get the index of the thread
  # there are as many threads as rows in the structure and each thread
  # computes the rankings corresponding to this and all the multiples according 
  # to the stride
  idx = cuda.grid(1)

  local_best_dist = np.inf

  if idx < stride: # otherwise accessing positions that cannot be written
    ranking_id = idx

    # the idx must be a permutation of the n candidates
    while ranking_id < total:

      # calculate the factoradic value
      quotient = ranking_id # initial number
      radix = 1
      while quotient != 0:
        quotient, remainder = divmod(quotient, radix)
        factorial_number[idx, n-radix]=remainder # write the panel
        radix+=1
      
      ############### GET THE RANKING FROM THE FACTORIAL NUMBER ##################

      # explored[idx,:] contains the alternatives that have/n't been explored
      # alternatives that have been already added to the
      # ranking during the exploration.
      # Initially all falso.
      # Important: boolean array to reduce memory
      # Como pudo ser usado previamente hay que ponerlo a false
      for i in range(n):
        alternatives[idx, i] = False
    
      # for each position of the factorial representation
      for i in range(n):
        # count Falses until:
        until = factorial_number[idx, i] + 1
        # logging.debug("Let's count {} False(s)".format(until))
        # set initial counter to 0
        count = 0
        # iterator for each position of the boolean alternatives
        alt = 0
        # iterate
        while count < until:
          if not alternatives[idx, alt]:
            count += 1
          alt += 1
        # at this point count = until and alt has the alternative 
        # that is in the i position
        # mark the alternative that appears in the ranking
        alternatives[idx, alt-1] = True
        # overwrite the ranking to save memory
        factorial_number[idx, i] = alt-1

      ############### GET THE DISTANCE TO THE PROFILE ############################
      dist = 0
      for i in range(n): # for each row
        for j in range(i+1, n): # for each col, consider only elements over diagonal
          if factorial_number[idx, i] > factorial_number[idx, j]:
            dist += profile[i,j]
          else: # factorial_number[i] < factorial_number[j] and also if the are the same 
            dist += profile[j,i]

      ############### UPDATE BEST DISTANCE #######################################

      if dist < local_best_dist:
        local_best_dist = dist

      # increment the stride to evaluate the next ranking
      ranking_id += stride

    # update the info in the
    # factorial_number[idx, 0] = local_best_dist