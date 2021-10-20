#!/bin/bash
#
# Run C++ code check. Formating and Linting.
#
# - Author: Jongkuk Lim
# - Contact: limjk@jmarple.ai


function run_cmd_on_source() {
    run_cmd=$1

    exitCode=0
    while read -d $'\0' file
    do
        eval >&2 "$run_cmd $file"

        retVal=$?

        if [ $retVal -ne 0 ]; then
            exitCode=$retVal
        fi
    done < <(find . -not -path './libs/*' -not -path './build/*' -regex '.*\.\(cpp\|hpp\|h\|cc\|cxx\)' -print0)

    echo "$exitCode"
}

CMD_FORMAT="clang-format -verbose -style=file -i"
CMD_LINT="cpplint"
# CMD_DOC_CHECK="( cat Doxyfile ; echo "GENERATE_HTML=NO"  ) | doxygen -"
CMD_DOC_CHECK="doxygen -q"
CMD_INIT_PRECOMMIT="pre-commit install --hook-type pre-commit --hook-type pre-push"

run_cmd=( "$CMD_FORMAT" )

if [ "$1" = "lint" ]; then
    run_cmd=( "$CMD_LINT" "cppcheck" )
elif [ "$1" = "format" ]; then
    run_cmd=( "$CMD_FORMAT" )
elif [ "$1" = "all" ]; then
    run_cmd=( "$CMD_FORMAT" "$CMD_LINT" "cppcheck" "doc_check" )
elif [ "$1" = "doc_check" ]; then
    run_cmd=( "doc_check" )
elif [ "$1" = "test" ]; then
    echo "Unit test(TBD)"
    exit 0
elif [ "$1" = "init-precommit" ]; then
    eval $CMD_INIT_PRECOMMIT
    exit $?
else
    echo ""
    echo "======== $0 [Usages] ========"
    echo "1) $0 format - run formating"
    echo "2) $0 lint - run linting check"
    echo "3) $0 all - run formating and linting"
    echo "4) $0 doc_check - run documentation check"
    echo "5) $0 init-precommit - install pre-commit config"

    exit 1
fi

### Formating and Linting
exitCode=0
for cmd in "${run_cmd[@]}"; do

    if [ "$cmd" = "cppcheck" ]; then
        cppcheck --quiet --inline-suppr --language=c++ --enable=all -I include --suppress=missingIncludeSystem --template "{file}({line}): {severity} ({id}): {message}" ./src ./tests
        result=$?
    elif [ "$cmd" = "doc_check" ]; then
        eval $CMD_DOC_CHECK
        result=$?
    else
        result=$(run_cmd_on_source "$cmd")
    fi

    if [ $result -ne 0 ]; then
        exitCode=$(($result))
    fi
done

exit $exitCode
