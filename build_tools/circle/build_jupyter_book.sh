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

apt-get install make

source /opt/conda/etc/profile.d/conda.sh
conda update --yes conda
conda create -n scikit-learn-mooc --yes -c conda-forge python=3.9
conda activate scikit-learn-mooc
pip install -r requirements-dev.txt

affected_jupyter_book_paths() {
    files=$(git diff --name-only origin/master...$CIRCLE_SHA1)
    # TODO: rather than the grep pattern below we could potentially look at
    # _toc.yml to know whether the file affects the JupyterBook
    echo "$files" | grep python_scripts | perl -pe 's@\.py$@.html@'
    echo "$files" | grep -P "$jupyter_book_dir/.+md$" | \
        perl -pe "s@$jupyter_book_dir/(.+)\.md@\1.html@"
}

write_changed_html() {
    affected="$1"
    if [ -n "$CI_PULL_REQUEST" ]
    then
        echo "The following files may have been changed by PR $CI_PULL_REQUEST:"
        echo "$affected"
        (
            echo '<html><body>'
            echo 'Files changed by PR <a href="'"$CI_PULL_REQUEST"'">'"$CI_PULL_REQUEST</a>"
            echo '<ul>'
            echo "$affected" | sed 's|.*|<li><a href="&">&</a> [<a href="https://inria.github.io/scikit-learn-mooc/&">master</a>]|'
            echo '</ul><p>This PR JupyterBook <a href="index.html">index</a>'
            echo '</ul></body></html>'
        ) > "$jupyter_book_build_dir/_changed.html"
    fi
}

affected=$(affected_jupyter_book_paths)

make $jupyter_book_dir 2>&1 | tee $jupyter_book_dir/build.log

write_changed_html "$affected"

# Grep the log to make sure there has been no errors when running the notebooks
# since jupyter-book exit code is always 0
grep 'Execution Failed' $jupyter_book_dir/build.log && show_error_logs || \
    echo 'All notebooks ran successfully'
