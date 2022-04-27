.POSIX:

container_engine=docker
# For podman first execute `printf 'unqualified-search-registries=["docker.io"]\n' > /etc/containers/registries.conf.d/docker.conf`
artifactsdir=artifacts
workdir=/app

user_arg=$(shell [ $(container_engine) = 'docker' ] && printf '%s' '--user `id -u`:`id -g`')

############################### code commands ##############################

.PHONY: $(artifactsdir)/code-requirements

codefile=main.py

debug_args=$(shell [ -t 0 ] && printf '%s' '--interactive --tty')
gpus_arg=$(shell [ $(container_engine) = 'docker' ] && command -v nvidia-container-toolkit > /dev/null && printf '%s' '--gpus all')

$(artifactsdir)/code-run: $(codefile) .dockerignore .gitignore Dockerfile requirements.txt ## 		Generate draft artifacts (FULL=1 for full).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env ARTIFACTSDIR=$(artifactsdir) \
		--env HOME=$(workdir)/$(artifactsdir) \
		--env FULL=$(FULL) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` python3 $(codefile)
	touch $(artifactsdir)/code-run

$(artifactsdir)/code-coverage: $(codefile) Dockerfile requirements.txt ## 	Code coverage for $(codefile).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys 'ctrl-^,ctrl-^' \
		--env ARTIFACTSDIR=$(artifactsdir) \
		--env HOME=$(workdir)/$(artifactsdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` /bin/bash -c '\
		coverage run $(codefile) && \
		coverage html'
	rm -rf $(artifactsdir)/htmlcov
	mv htmlcov/ $(artifactsdir)/
	mv .coverage $(artifactsdir)/
	touch $(artifactsdir)/code-coverage)

$(artifactsdir)/code-format: $(codefile) ## 	Format $(codefile).
	mkdir -p $(artifactsdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` /bin/bash -c '\
		python-minimizer --keep-whitespace --out-path $(codefile) $(codefile) && \
		isort $(codefile) && \
		autoflake --expand-star-imports --in-place --remove-all-unused-imports --remove-duplicate-keys --remove-unused-variables $(codefile) && \
		pyupgrade $(codefile) && \
		autopep8 -a -a -a --in-place --max-line-length 10000 $(codefile)'
	touch $(artifactsdir)/code-format

$(artifactsdir)/code-requirements: ## 	Generate $(codefile) requirements.txt.
	mkdir -p $(artifactsdir)/
	rm -f requirements.txt
	make requirements.txt
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` /bin/bash -c 'pipreqs --print . >> requirements.txt'

$(codefile):
	printf "import os\n\n\ndef main():\n    artifacts_dir = os.getenv('ARTIFACTSDIR')\n    full = os.getenv('FULL')\n\n\nif __name__ == '__main__':\n    main()\n" > $(codefile)

Dockerfile:
	printf 'FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n' > Dockerfile

requirements.txt:
	printf '# Makefile requirements\nautoflake\nautopep8\ncoverage\nisort\npipreqs\npython-minimizer\npyupgrade\n\n# $(codefile) requirements\n' > requirements.txt

############################### Makefile commands ##############################

.PHONY: clean help

clean: ## 			Remove artifacts/ directory.
	rm -rf $(artifactsdir)/

help: ## 				Show all commands.
	@grep '##' $(MAKEFILE_LIST) | sed 's/\(\:.*\#\#\)/\:\ /' | sed 's/\$$(artifactsdir)/$(artifactsdir)/' | sed 's/\$$(codefile)/$(codefile)/' | sed 's/\$$(texfile)/$(texfile)/' | grep -v grep

.dockerignore:
	printf '**\n**/.*\n!requirements.txt\n' > .dockerignore

.gitignore:
	printf '$(artifactsdir)/\n' > .gitignore
