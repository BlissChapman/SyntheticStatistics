# Synthetic Statistics
This project explores the application of generative modeling techniques to statistical testing.

### Objective
Test whether two distributions P and Q are different based on samples drawn from each of them.

### Parametric Approach
1) Sample from distribution P.
2) Sample from distribution Q.
3) Run a classical two sample t-test.

But why assume the data is Gaussian?

### Proposed Non-Parametric Approach
1) Given samples from P, use a generative model to recover an estimate P^ of the underlying probability distribution of P.
2) Given samples from Q, use a generative model to recover an estimate Q^ of the underlying probability distribution of Q.
3) Using unlimited synthetic data from the learned distributions P^ and Q^, run a non-parametric test like a [Kernel Two-Sample Test](http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm) to test for a difference between P^ and Q^.

The goal of this project is to show that recovering the underlying data distribution can enable powerful non-parametric tests that take advantage of unlimited synthetic data.  Hopefully this leads to more accurate experimental power estimates and better research :D

## Experiments (MORE TO COME)
1. Fit two GANs to two different exponential distributions. Run an MMD test and a two sample t-test and compare the results. Repeat for two identical exponential distributions.

## Neuroscientific Application
> Mounting evidence over the last few years suggest that published neuroscience research suffer from low power, and especially
> for published fMRI experiments. Not only does low power decrease the chance of detecting a true effect, it also reduces the
> chance that a statistically significant result indicates a true effect ([Ioannidis, 2005](http://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124)). Put another way, findings with the
> least power will be the least reproducible, and thus a (prospective) power analysis is a critical component of any paper.

> In a scientific study, one typically aims for a statistical power of 80%, implying that a true effect in the population is
> detected with a 80% chance. Power computations allow researchers to compute the minimal number of subjects to obtain the
> aimed statistical power. As such, power calculations avoid spending time and money on studies that are futile, and also
> prevent wasting time and money adding extra subjects, when sufficient power was already available.

Source: [Power and sample size calculations for fMRI studies based on the prevalence of active peaks](https://www.biorxiv.org/content/biorxiv/early/2016/04/20/049429.full.pdf)
