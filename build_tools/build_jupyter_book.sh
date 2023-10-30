#!/usr/bin/env bash
set -xe

jupyter_book_dir=jupyter-book
jupyter_book_build_dir="$jupyter_book_dir/_build/html"

function show_error_logs {
    echo "Some notebooks failed, see logs below:"
    for f in $jupyter_book_build_dir/reports/*.log; do
        echo "================================================================================"
        echo $f
        echo "================================================================================"
        cat $f
    done
    # You need to exit with non-zero here to cause the build to fail
    exit 1
}

affected_jupyter_book_paths() {
    files=$(git diff --name-only origin/main...)
    # TODO: rather than the grep pattern below we could potentially look at
    # _toc.yml to know whether the file affects the JupyterBook
    echo "$files" | grep python_scripts | perl -pe 's@\.py$@.html@'
    echo "$files" | grep -P "$jupyter_book_dir/.+md$" | \
        perl -pe "s@$jupyter_book_dir/(.+)\.md@\1.html@"
}

write_changed_html() {
    affected="$1"
    if [ -n "$GITHUB_PULL_REQUEST_NUMBER" ]
    then
        GITHUB_PULL_REQUEST_URL="https://github.com/inria/scikit-learn-mooc/pull/$GITHUB_PULL_REQUEST_NUMBER"
        echo "The following files may have been changed by PR $GITHUB_PULL_REQUEST_NUMBER:"
        echo "$affected"
        (
            echo '<html><body>'
            echo "Files changed by PR <a href=\"$GITHUB_PULL_REQUEST_URL\">$GITHUB_PULL_REQUEST_URL</a>"
            echo '<ul>'
            echo "$affected" | sed 's|.*|<li><a href="&">&</a> [<a href="https://inria.github.io/scikit-learn-mooc/&">main</a>]|'
            echo '</ul><p>This PR JupyterBook <a href="index.html">index</a>'
            echo '</ul></body></html>'
        ) > "$jupyter_book_build_dir/_changed.html"
    else
        echo "The variable 'GITHUB_PULL_REQUEST_NUMBER' is not defined: not writing the '_changed.html' file."
    fi
}

git remote -v
git show --stat
git log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit -20
git fetch origin main >&2 # || (echo QUICK BUILD: failed to get changed filenames for $git_range; return)
git diff origin/main... --stat
git diff origin/main...

affected=$(affected_jupyter_book_paths)
mkdir -p $jupyter_book_build_dir
write_changed_html "$affected"

make $jupyter_book_dir 2>&1 | tee $jupyter_book_dir/build.log


# Grep the log to make sure there has been no errors when running the notebooks
# since jupyter-book exit code is always 0
grep 'Execution Failed' $jupyter_book_dir/build.log && show_error_logs || \
    echo 'All notebooks ran successfully'
