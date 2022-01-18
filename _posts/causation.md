---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Causation may not imply correlation


Much of our scientific and everyday understandings of the world depend heavily on the notion that cause and effect reliably co-vary. When we try to elucidate causal relationships between variables we usually try to understand how a state change in one variable induces a state change in another. Explicating these relationships can be quite different. Correlation can be a valuable tool in this kind of inquiry. It is sometimes used as a measuring stick to ask "is there something going on here?" And in many cases, it serves as an appropriate tool.  Without the right conditions in place, however, correlation is insufficient for causal inference. Indeed, under some conditions, correlation finds relationships where there are none, falsely answering "yes" to the "is there something going on here" question. On the other hand, correlation may miss the mark on occasion when asking the same question and falsely identify no relationship when one is present. While much attention has been brought to the former case (you can get started [here](https://statmodeling.stat.columbia.edu/2014/08/04/correlation-even-imply-correlation/)), the latter case is what I'd like to focus on today. To adapt some wise words from Samuel L. Jackson, "the absence of [correlation] is not the evidence of absence [of an influential relationship/coupling among variables]." 

One area of study that I'd like to draw attention to here is the field of resting-state functional MRI in neuroscience. The main idea here is to observe endogenous brain activity of subjects while they lie still in the scanner for a prolonged period. During this time, the scanner collects a brain scan every couple of seconds, after which the researchers are left with a dataset composed of a high-dimensional time series (many spatial measurements by  many time points). To see which brain regions interact most, it is typical to correlate the time series of different brain regions. This and other methods have resulted in elucidating a number of interesting relationship including the "default-mode network," the "salience network," among many others. This kind of analysis is sometimes called "functional connectivity analysis" and it looks to shed light on dependencies among areas of the brain.

What I'd like to draw attention to here is that while a correlation among brain areas may indicate an interesting relationship, the opposite claim that the absence of correlation indicates that there is no interesting relationship is unwarrented and misleading. This inference follows all too easily when we use loaded terminology like "functional connectivity," with the implication that little or no correlation indicates that 1) these regions are not literally connected (not a safe inference) and 2) there is no communication/influence between these regions (also not a safe inference). It is for this reason that I find the adoption of the term "connectivity" to refer to time series correlations unfortunate. In conversations with collegues, many have been under the impression that functional connectivity implies a physical, structural connection and that the absence of functional connectivity between brain regions implies that there is no interesting interactions or communications between these regions.    


To highlight how important relationships can be masked when correlation is our only tool, I'd like to examine an entirely hypothetical dynamic that is in no way based on personal experience. Suppose there are two people: Dale and Girl, which we'll denote with D and G respectively in the below equations. Let's suppose that Dale and Girl are romantically interested in one another. Dale, being a sensible young man, is drawn to Girl when she expresses interest in him. His interest wanes, however, when she is cold and distant. Girl, on the other hand, finds Dale's interest, kind words, and actions to be boring, and thus her interest in him wanes when his interest is mutual. When Dale shows less interest, though, Girl finds him strangly mysterious and attractive, or something. We can express this dynamic in the following way:

$\dot{D} = G$

$\dot{G} = -D$

The time derivative of Dale's interest ($\dot{D}$) is equal to Girl's current interest. The time derivative of Girl's interest ($\dot{G}$) is equal to minus Dale's current interest. These are simple linear dynamics and we can use a matrix-vector-product to represent them:

$
\begin{bmatrix} \dot{D} \\ \dot{G} \end{bmatrix}
 =
  \begin{bmatrix}
   0 & 1 \\
   -1 & 0
   \end{bmatrix}
   \begin{bmatrix}
   D \\ G
   \end{bmatrix}
$

The purely imaginary eigenvalues of this matrix tell us that the functions D(t) and G(t) are out-of-phase sinusoids.

$\lambda = \alpha + i\omega$

$\lambda_1 = 0 + i, \lambda_2 = 0 - i$


<!-- #endregion -->

```python
import numpy as np
import matplotlib.pyplot as plt
import findiff
import pysindy as ps
from scipy.integrate import odeint
from scipy.stats import pearsonr
```

```python
A = np.array([[0,1],[-1,0]])
eigenvalues = np.linalg.eigvals(A)
eigenvalues
```


We can integrate the dynamics forward in time and confirm that this is the case:

```python
def girl_problem(state, params=None):
    return A.dot(state)

t = np.linspace(0,30,3000)
out = odeint(girl_problem, [.5,.5], t)

ax = plt.subplot()

ax.plot(out[:,0])
ax.plot(out[:,1])
ax.set_xlabel('time')
ax.set_ylabel('romantic interest')
ax.legend(['D(t)', 'G(t)'])
```

We can also look at the vector field by plotting G and D against each each:

```python
fig, axs = plt.subplots(1,2, figsize=(10,5))
x,y = np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))

u = y
v = -x
axs[0].quiver(x,y,u,v)
axs[0].scatter(0,0)
axs[0].set_xlabel('G')
axs[0].set_ylabel('D')

dynamics = [odeint(girl_problem, cond, t) for cond in [[.5,.5], [.7,.7],[1,1], [1.5,1.5]]]

for out in dynamics:
    axs[1].plot(out[:,0], out[:,1])
axs[1].scatter(0,0)
axs[1].set_xlabel('G(t)')
axs[1].set_ylabel('D(t)');

```

We call this kind of dynamics a center. This does not bode well for Dale and Girl. This matrix simply rotates the current state around the origin for eternity. However interested Dale and Girl are at one particular moment, they will find themselves in that same place after some highs and lows on the loop around the origin.


When we take the correlation between these two time series, we get a correlation close to zero, as we might expect for out-of-phase sinusoids:

```python
pearsonr(out[:,0], out[:,1])[0]
```

Despite these two agents exerting mutual influence on one another, this relationship is not reflected in the r value. Correlation simply misses the mark for answering the "is there something going here" question.

<!-- #region -->
This is where Dale turns to his friend Resting State MRI Enthusiust (RSMRIE for short) for help. RSMRIE takes some careful observations of the dynamics between Dale and Girl and collects two vectors of data corresponding to the time series for each person. RSMRIE analyses the data, but the results do not please her. With much chagrin, she tells Dale the bad news:

RSMRIE: "Dale, I know that you've been saying that you've had a crazy up-and-down experience with Girl, but I've run the numbers. There's no correlation in your time series. I mean, do you even know this girl?

Dale: "That can't be. It's been a roller coaster. We've causally affected each other in clear and substantive ways."

RSMRIE: "With an r value of -0.0053698497298204795 I wouldn't talk about correlation let alone causation. Sorry, big guy."


All of the sassiness aside, there are a number of lessons we can learn from this hypothetical, in-no-way-real story.

- It would be silly and inaccurate to say that these two individuals are not influencing each others' behavior. Because of the properties of each person, the resulting dynamic occurs without any correlation among the observable variables. The absence of correlation does not imply the absence of coupling.
- This example features a two-dimensional linear system. Real recordings of brain activity are high dimensional, likely nonlinear, and have latent sources of variance. Just as correlation misses the mark in this example, it is likely to miss many important interactions and relationships in the brain.

The prevailing methods of resting state analysis have highlighted interesting aspects of brain activity. And I don't mean to suggest otherwise. Like any method of analysis, we should be cautious about the inferences that we let ourselves make from the results that we observe. Especially when studying a complicated dynamical system like the brain, there is no reason to expect correlation to definitively answer the "is there something going on here" question. Once more, the absence of correlation does not imply the absence of coupling.

None of this is to say that correlation-based methods cannot be useful or shouldn't be used. Singular value decomposition-based methods like the Proper Orthogonal Decomposition (POD) have been widely used in fluid dynamics for a long time now with great success. Like the brain, these are high-dimensional dynamical systems and POD is a means of performing dimensionality reduction based on correlations among the observed variables.

A reasonable question to ask at this point is aside from making more cautious inferences and avoiding misleading terminology, are there methods to identify coupled variables in the absence of correlation? The above very fictional story can give us some insight here. The important relationships here have to do with how the changes of D evolve as a function of G and how the changes of G evolve as a function of -D. To get at this relationship, we have look at some derivatives.



<!-- #endregion -->

```python
d_dt = findiff.FinDiff(0, 0.01000333)
df_dt = d_dt(dynamics[0])
print("correlation between dD/dt and G(t): ", pearsonr(df_dt[:,0], dynamics[0][:,1])[0])
print("correlation between dG/dt and D(t): ", pearsonr(df_dt[:,1], dynamics[0][:,0])[0])
```

With those correlations, RSMRIE would be pleased, at least with the first one.

By examining the derivatives like this, we have done more than identify a strong relationship between measurements. These two correlations give us all we need to reconstruct the dynamics of how these observations were generated. If we round these values to 1 and -1 we can get our original matrix:

$
\begin{bmatrix} \dot{D} \\ \dot{G} \end{bmatrix}
 =
  \begin{bmatrix}
   0 & 1 \\
   -1 & 0
   \end{bmatrix}
   \begin{bmatrix}
   D \\ G
   \end{bmatrix}
$

We intuitively pieced this model back together by correlating derivatives with our observations. We can do better than this. Using the SINDy algorithm, we can learn a complete model from the data with out doing the comparisons ourselves.


```python
model = ps.SINDy(feature_names=['D', 'G'])
model.fit(out, t)
model.print()
```

<!-- #region -->
This SINDy model was able to learn the dynamics exactly. SINDy works by doing a simple regression $AX = B$ where $A$ is our data with each column representing a different variable, $B$ holds the time derivatives of our data, and X is the latent matrix of coefficients. To identify nonlinear dynamics, we include products and polynomial functions of our original variables. And to make sure that we generate as few coefficients as possible to fit the data, we use a sparsity-promoting regression optimization. For our data at hand, it might looks something like this:



$
\begin{bmatrix} \dot{D} & \dot{G} \\ \vdots & \vdots \end{bmatrix} = 
\begin{bmatrix} 1 & D & G & DG & D^2 & G^2 & D^3 & G^3 \\
                \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}
\begin{bmatrix} 0 & 0\\
                0 & -1\\
                1 & 0\\
                0 & 0\\
                0 & 0\\
                0 & 0\\
                0 & 0\\
                0 & 0\\
\end{bmatrix}
$

Using this kind of technique, we're able to generate data-driven dynamical models even in the presence of some noise. Such an approach does not scale well to high dimensional data like most neuroimaging data unless dimensionality reduction of some sort is applied.

Another relevent approch, the Dynamic Mode Decomposition (DMD), also uses a regression to learn the dynamics of a given dataset. DMD seeks to learn the eigen decomposition of the best discrete-time linear operator that maps any given time point onto the next point (best in a least squares sense). DMD is well-suited to the high-dimensional data of neuroimaging.

What these approaches have in common is that they both model the data at hand as a dynamical system. They model the underlying principles of interaction that generate the data. This is the dominant approach in theoretical/computational neuroscience. I think that there are a number of reasons as to why wildy different approaches are adopted in different areas of neuroscience. I suspect it may have to do with different kinds of questions being asked, differing levels in stimulus and measurement complexity, differing educational backgrounds of researchers among the different subfields, computational tractability, and different noise magnitudes, among other things. In the case of resting-state fMRI, as well as passive video watching fMRI data, I think there is potential to augment the current set of tools and improve theoretical frameworks by adopting a dynamical systems approach.

In this brief essay, it has been my aim to convey a few points that I think are important:
- Causal relationships need not manifest as correlation among the observed variables.
- Echoing the previous point, the absence of correlation does not imply the absence of coupling between variables.
- "Functional connectivity" just means correlated. Just as "statistically significant" doesn't mean "of great interest" or "important." It just means p < .05. Just as "confidence intervals" have nothing to do with confidence. The common-sense inferences from what these terms mean in english don't apply well to many domain-specific terms. Now, I suspect some readers will point to the literature showing that some brain areas that are structurally well connected to each other can be correlated with each other in time sometimes. Nothing that I've said contradicts that. Simply put, inferring connectivity from [correlation is not safe](https://jon-e.net/blog/2019/03/12/Correlation-Isnt-Connectivity/). And especially because correlation is insensitive to most kinds of dynamical coupling, particularly the non-linear dynamics of the brain, I think we shouldn't use the term "connectivity" at all in these contexts.

Whew, alright let's keep going:
- Approches like SINDy and DMD can help to elucidate the dynamics of our data.
- Functional neuroimaging can augment their set of tools and improve their theoretical frameworks by adopting a dynamical systems approach, at least in resting-state data, if not elsewhere as well.

I demonstrated these points with an example that couldn't possibly be based on a true story.



<!-- #endregion -->

```python

```
