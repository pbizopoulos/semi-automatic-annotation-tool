.POSIX:

container_engine=docker
# For podman first execute `echo 'unqualified-search-registries=["docker.io"]' > /etc/containers/registries.conf.d/docker.conf`
tmpdir=tmp
workdir=/app

user_arg=$(shell [ $(container_engine) = "docker" ] && echo --user `id -u`:`id -g`)

# Basic commands.

.PHONY: clean

pythonfile=main.py

debug_args=$(shell [ -t 0 ] && echo --interactive --tty)
gpus_arg=$(shell [ $(container_engine) = "docker" ] && command -v nvidia-container-toolkit > /dev/null && echo --gpus all)

$(tmpdir)/python-run: $(pythonfile) .dockerignore .gitignore Dockerfile requirements.txt
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--env FULL=$(FULL) \
		--env TMPDIR=$(tmpdir) \
		--publish 5000:5000 \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` python3 $(pythonfile)
	touch $(tmpdir)/python-run

clean:
	rm -rf $(tmpdir)/

help:
	@echo "Basic/Advanced commands:                                   "
	@echo "                         # Basic commands.                 "
	@echo "make                     # Generate draft (fast) results.  "
	@echo "make FULL=1              # Generate full (slow) results.   "
	@echo "make clean               # Remove tmp/ directory.          "
	@echo "make help                # Show basic/advanced commands.   "
	@echo "                         # Advanced commands.              "
	@echo "                         # python commands.                "
	@echo "make tmp/python-coverage # Code coverage for main.py.      "
	@echo "make tmp/python-format   # Format main.py.                 "

$(pythonfile):
	printf "import os\n\ntmpdir = os.getenv('TMPDIR')\nfull = os.getenv('FULL')\n\n\ndef main():\n    pass\n\n\nif __name__ == '__main__':\n    main()\n" > $(pythonfile)

.dockerignore:
	printf ".git/\ntmp/\n" > .dockerignore

.gitignore:
	printf "tmp/\n" > .gitignore

Dockerfile:
	printf "FROM python\nCOPY requirements.txt .\nRUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt\n" > Dockerfile

requirements.txt:
	printf "# Makefile requirements\nautoflake\nautopep8\ncoverage\nisort\n\n# document requirements\n\n" > requirements.txt

# python commands.

$(tmpdir)/python-coverage: $(pythonfile) Dockerfile requirements.txt
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(debug_args) \
		$(gpus_arg) \
		$(user_arg) \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir) \
		--env TMPDIR=$(tmpdir) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` bash -c "coverage run $(pythonfile) && coverage html && rm -rf $(tmpdir)/htmlcov && mv htmlcov/ $(tmpdir)/ && mv .coverage $(tmpdir)/"
	touch $(tmpdir)/python-coverage

$(tmpdir)/python-format: $(pythonfile)
	mkdir -p $(tmpdir)/
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`$(container_engine) image build --quiet .` bash -c "isort $(pythonfile) && autoflake --in-place --remove-all-unused-imports --remove-unused-variables $(pythonfile) && autopep8 -i --max-line-length 1000 $(pythonfile)"
	touch $(tmpdir)/python-format
