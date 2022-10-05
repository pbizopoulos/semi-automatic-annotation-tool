.POSIX:

.PHONY: clean help

artifacts_dir=artifacts
code_file_name=index.js

$(artifacts_dir)/code-run: $(artifacts_dir) $(code_file_name) .gitignore docs/* package-lock.json ## Generate artifacts.
	ARTIFACTS_DIR=$(artifacts_dir) node $(code_file_name)
	touch $(artifacts_dir)/code-run

clean: ## Remove dependent directories.
	rm -rf $(artifacts_dir)/ node_modules/ package-lock.json

help: ## Show all commands.
	@sed -n '/sed/d; s/\$$(artifacts_dir)/$(artifacts_dir)/g; /##/p' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.* ## "}; {printf "%-30s# %s\n", $$1, $$2}'

$(artifacts_dir):
	mkdir $(artifacts_dir)

$(code_file_name):
	printf '\n' > $(code_file_name)

.gitignore:
	printf '$(artifacts_dir)/\nnode_modules/\npackage-lock.json\n' > .gitignore

package-lock.json: package.json
	npm install

package.json:
	printf '{}\n' > package.json
