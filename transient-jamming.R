num_channels <- 16
rounds <- 1000
time_horizon <- 1000
cumulative_rewards <- vector(length = rounds)

generate_pdrs <- function(target_mean_pdr = 0.5) {
  repeat {
    pdrs <- runif(num_channels)
    best_channel <- sample(seq(num_channels), size = 1)
    pdrs[best_channel] = 1.0
    discrepancy <- target_mean_pdr - mean(pdrs)
    pdrs <- pdrs + discrepancy + (discrepancy / (num_channels - 1))
    pdrs[best_channel] <- 1.0
    if((min(pdrs) >= 0) && (max(pdrs) <= 1)) {
      return(pdrs)
    }
  }
}

generate_reward <- function(t,
                            channel,
                            pdrs,
                            with_transient_jamming = TRUE,
                            jamming_start = 500,
                            jamming_end = 599) {
  if(!with_transient_jamming
      || (t < jamming_start)
      || (t > jamming_end)) {
    return(rbinom(1, 1, pdrs[channel]))
  }
  
  if(pdrs[channel] < 0.9) {
    return(rbinom(1, 1, pdrs[channel]))
  }
  # transient jamming
  return(0)
}

results <- c()
for(with_transient_jamming in c(FALSE, TRUE)) {
  for(round in seq(rounds)) {
    cumulative_rewards[round] <- 0
    pdrs <- generate_pdrs()
    
    # initialize SW-UCB
    selections <- rep(0, num_channels)
    window_size <- 1000
    exploration_tendency <- 2**-13
    sliding_windows <- matrix(nrow = num_channels, ncol = window_size)
    
    for(t in seq(time_horizon)) {
      if(0 %in% selections) {
        channel <- match(0, selections)
      } else {
        ucbs <- rep(0, num_channels)
        for(c in seq(num_channels)) {
          ucbs[c] <- mean(sliding_windows[c,][1:selections[c]]) + sqrt(exploration_tendency * min(c(t - 1, window_size)) / selections[c])
        }
        channel <- which.max(ucbs)
      }
      
      # generate reward
      reward <- generate_reward(t, channel, pdrs, with_transient_jamming)
      cumulative_rewards[round] <- cumulative_rewards[round] +  reward
      
      # update variables
      selections[channel] <- selections[channel] + 1
      if(selections[channel] > window_size) {
        for(i in 2:window_size) {
          sliding_windows[channel, i - 1] <- sliding_windows[channel, i]
        }
        sliding_windows[channel, window_size] <- reward
      } else {
        sliding_windows[channel, selections[channel]] <- reward
      }
    }
  }
  results <- c(results, mean(cumulative_rewards))
  
  for(round in seq(rounds)) {
    cumulative_rewards[round] <- 0
    pdrs <- generate_pdrs()
    
    # initialize gradient bandit
    baselines <- rep(0.5, num_channels)
    selections <- rep(1, num_channels)
    alpha_gradient <- 0.9
    preferences <- rep(0, num_channels)
    stationary <- FALSE
    alpha_non_stationary <- 0.9
    
    for(t in seq(time_horizon)) {
      # select channel
      probabilities <- vector(length = num_channels)
      sum_of_exps <- sum(exp(preferences))
      for(c in seq(num_channels)) {
        probabilities[c] = exp(preferences[c]) / sum_of_exps
      }
      channel <- sample(seq(num_channels), size = 1, prob = probabilities)
      
      # generate reward
      reward <- generate_reward(t, channel, pdrs, with_transient_jamming)
      cumulative_rewards[round] <- cumulative_rewards[round] +  reward
      
      # update variables
      selections[channel] <- selections[channel] + 1
      if (stationary) {
        baselines[channel] <- baselines[channel] + ((1/selections[channel]) * (reward - baselines[channel]))
      } else {
        baselines[channel] <- baselines[channel] + (alpha_non_stationary * (reward - baselines[channel]))
      }
      for(c in seq(num_channels)) {
        if (c == channel) {
          preferences[c] <- preferences[c] + ((alpha_gradient * (reward - baselines[c])) * (1 - probabilities[c]))
        } else {
          preferences[c] <- preferences[c] - ((alpha_gradient * (reward - baselines[c])) * probabilities[c])
        }
      }
    }
  }
  results <- c(results, mean(cumulative_rewards))
  
  for(round in seq(rounds)) {
    cumulative_rewards[round] <- 0
    pdrs <- generate_pdrs()
    
    # initialize D-UCB
    discount_factor <- 1 - 2**-10
    exploration_tendency <- 2**-13
    n <- rep(0, num_channels)
    x <- rep(0, num_channels)
    
    for(t in seq(time_horizon)) {
      # select channel
      if (t <= num_channels) {
        channel <- t
      } else {
        ducbs <- vector(length = num_channels)
        for(k in seq(num_channels)) {
          ducbs[k] <- x[k] / n[k] + sqrt(exploration_tendency * log(sum(n)) / n[k])
        }
        channel <- which.max(ducbs)
      }
      
      # generate reward
      reward <- generate_reward(t, channel, pdrs, with_transient_jamming)
      cumulative_rewards[round] <- cumulative_rewards[round] +  reward
      
      # update variables
      n <- n * discount_factor
      x <- x * discount_factor
      n[channel] <- n[channel] + 1
      x[channel] <- x[channel] + reward
    }
  }
  results <- c(results, mean(cumulative_rewards))
}
barplot(results,
        names = c("SW-UCB/no jamming",
                  "gradient bandit/no jamming",
                  "D-UCB/no jamming",
                  "SW-UCB/jamming",
                  "gradient bandit/jamming",
                  "D-UCB/jamming"),
        ylab="mean cumulative reward")

print("SW-UCB")
print(results[4] * 100 / results[1])
print("gradient bandit")
print(results[5] * 100 / results[2])
print("D-UCB")
print(results[6] * 100 / results[3])