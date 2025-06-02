# MOOC workflow notes

## Github repo

Most of the public MOOC content is in the github repo:
https://github.com/INRIA/scikit-learn-mooc. The rendered version of the
`main` branch is at: https://inria.github.io/scikit-learn-mooc.

Note that some files in this repo are derived files, meaning you are not
supposed to edit them directly but to generate them.

In particular We use `.py` files in the `python_scripts` folder for easier
version control and generate the `.ipynb` from the `.py` files.

The following table is a summary of the derived files and how to generate them.

| derived files                              | source files                                | command to generate |
| ------------------------------------------ | ------------------------------------------- | ------------------- |
| `notebooks/<filename>.ipynb`               | `python_scripts/<filename.py>`              | `make notebooks`    |
| `python_scripts/<filename>_ex_<number>.py` | `python_scripts/<filename>_sol_<number.py>` | `make exercises`    |
| `jupyter-book/<folder>/<filename>_quiz.py` | same file in the gitlab repo                | `make quizzes`      |

To see how github repo changes are taken into account in the FUN-MOOC platform,
see [below](#how-our-repo-contents-are-used-on-the-fun-mooc-platform).

### Continuous Integration

We use Github Actions for:
- building the JupyterBook in pull requests. Previewing the built JupyterBook
  is done by deploying to Netlify.
- building the JupyterBook on pushes to main and deploy to gh-pages

### Label conventions with the Learning Lab

Github labels:
- `FUN:requires change` and `FUN:action done` were used to indicate in an
  issue. Laurent and Marie were looking at the closed issues with `FUN:
  requires change` from time to time when we were developing the material

## Gitlab repo

The quizzes solutions can not be public so they are in our private repo
https://gitlab.inria.fr/learninglab/mooc-scikit-learn/mooc-scikit-learn-coordination.

The `jupyter-book` folder has the same structure as the `jupyter-book` folder
on the github repo, but the gitlab one contains only quizzes `.md` files. If
you work on quizzes, you need to do it in the gitlab repo, the github repo
quiz files are generated from the gitlab repo (by stripping solution) with
`make exercises`.

Useful: to get the `.py` code from a quiz `.md`, look at
./CONTRIBUTING.md#get-wrap-up-quiz-solutions-code

## How our repo contents are used on the FUN-MOOC platform

We try to work as much as possible on our repo because this is the way we are
used to, like, and feel we work efficiently. At one point some manual update
needs to be done in FUN inside the FUN-MOOC studio which is the kind of
click-heavy interface that as developers tends to make us feel frustrated quite
fast.

### Exercise correction, module overview, take-away message, glossary, ...

They use some .html file in our github.io.

The `remove-from-content-only` CSS class is used to remove content that should
only be in our github.io but not in FUN-MOOC, for example the left-hand side
panel, footer, etc ...

There are two ways this can be applied on the FUN side:
- adding `?content_only` at the end of a URL, for example
  https://inria.github.io/scikit-learn-mooc/?content_only (compare with
  https://inria.github.io/scikit-learn-mooc/). The javascript logic comes from
  `jupyter-book/_static/sklearn_mooc.js`.
- manually including HTML with some javascript magic on the FUN-MOOC side e.g.
  the concluding remarks that excludes content based on CSS classes.
  ```html
  <div class="external-resource"
       data-hide="h1,.topbar, .prev-next-area, .footer, .site-navigation, .headerlink, .remove-from-content-only"
       data-url="https://inria.github.io/scikit-learn-mooc/concluding_remarks.html">
  ```
  Note that in this case they can not use `?content_only` because this method is based on some js
  code that loads the HTML without loading the js in sklearn_mooc.js. This means there is some
  duplication of logic between data-hide and sklearn_mooc.js (e.g. to remove navigation items from
  JupyterBook) but oh well 🤷‍♂️ ... the best we can do is to use the `remove-from-content-only`
  class in JupyterBook.

### Notebooks

Note: FUN use notebooks so if you only update the `.py` files, FUN participants
will not see the changes. Also: they need to manually "reset to original" their
notebook, see
https://mooc-forums.inria.fr/moocsl/t/how-to-reset-a-notebook-to-its-original-version/381/2.

Notebooks in FUN is an iframe opening a notebook on JupyterHub with the right
notebook path.

Note that some notebook changes need manual action in FUN:

- adding a notebook or moving an existing notebook in a separate lesson or
  module (i.e. any change to `jupyter-book/_toc.yml`)
- renaming a `.ipynb` or `.md` filename (this likely needs a
  `jupyter-book/_toc.yml` anyway as in the previous bullet point)
- changing a quizz. Quizzes are updated manually, the Learning Lab looks at our
  markdown and either writes some markdown-like thing for simple quizzes or
  write some hand-crafted HTML to look like our quiz. Sometimes they also
  decide to use some FUN specific thing like using FUN hint when we put "Hint:"
  in the markdown. Not sure that is a great idea in itself but oh well ...

### Empty wrap-up quiz notebooks and sandbox notebook

Wrap-up quiz notebooks or sandbox notebooks are created on the FUN side. In other words, we do not have
an empty notebook for each wrap-up quiz in our github repo.

## Discourse forum

- tags: `priority-*` to indicate we should do it and how much we care about it.
  Currently it is fair to assume everything has roughly the same priority i.e.
  `priority-mooc-v2`. `priority-nice-to-hav` may indicate some not so important
  thing (but not 100% guaranteed).
- tags: `fun-needs-action` when something needs to be updated in FUN,
  `fun-action-done` when it is done.

## Miscellaneous

Special thing: when committing something in gitlab Laurence or Marie will pick
it up (they receive an email I think) and tackle it, if this is a quiz change.
They tend to put a comment in the commit, honestly this should be improved for
better tracking since we are not seeing the message on gitlab ... github issue
with manual link to the gitlab commit ? gitlab issue/or MR with label (with the
risk of having two issue trackers ...) ?
