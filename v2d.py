@cuda.jit
def v2d(factorial_number, alternatives, profile, stride, total, best_dist, best_ranking):
  
  ############### GET THE FACTORIAL NUMBER FROM IDX ############################

  # get the number of elements
  n = factorial_number.shape[1]

  # get the index of the thread
  # there are as many threads as rows in the structure and each thread
  # computes the rankings corresponding to this and all the multiples according 
  # to the stride
  idx = cuda.grid(1)

  local_best_dist = np.inf
  best_id = total+1

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

            if dist > local_best_dist:
                break
        break

      ############### UPDATE BEST DISTANCE #######################################

      if dist < local_best_dist:
        local_best_dist = dist
        best_id = ranking_id

      # increment the stride to evaluate the next ranking
      ranking_id += stride

    # if local_best_dist < best_dist: 
    #   best_dist = local_best_dist (this must be done with atomic access)
    value = cuda.atomic.min(best_dist, 0, local_best_dist)
    # value contains the value of the min
    if value > local_best_dist:
      best_ranking[0] = ranking_id