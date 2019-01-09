# -*- coding: utf-8 -*-
"""
Sphinx_ backend for building docs.

.. _Sphinx: http://www.sphinx-doc.org/
"""
from __future__ import (
    absolute_import, division, print_function, unicode_literals)

import codecs
import shutil
import logging
import os
import sys
import zipfile
from glob import glob

import six
from django.conf import settings
from django.template import loader as template_loader
from django.template.loader import render_to_string

from readthedocs.builds import utils as version_utils
from readthedocs.projects.exceptions import ProjectConfigurationError
from readthedocs.projects.utils import safe_write
from readthedocs.restapi.client import api
from readthedocs.projects.models import Feature

from ..base import BaseBuilder, restoring_chdir
from ..constants import PDF_RE
from ..environments import BuildCommand, DockerBuildCommand
from ..exceptions import BuildEnvironmentError
from ..signals import finalize_sphinx_context_data

log = logging.getLogger(__name__)


class BaseSphinx(BaseBuilder):

    """The parent for most sphinx builders."""

    def __init__(self, *args, **kwargs):
        super(BaseSphinx, self).__init__(*args, **kwargs)
        self.config_file = self.config.sphinx.configuration
        try:
            if not self.config_file:
                self.config_file = self.project.conf_file(self.version.slug)
            self.old_artifact_path = os.path.join(
                os.path.dirname(self.config_file),
                self.sphinx_build_dir
            )
        except ProjectConfigurationError:
            docs_dir = self.docs_dir()
            self.old_artifact_path = os.path.join(
                docs_dir,
                self.sphinx_build_dir,
            )

    def _write_config(self, master_doc='index'):
        """Create ``conf.py`` if it doesn't exist."""
        docs_dir = self.docs_dir()
        conf_template = render_to_string(
            'sphinx/conf.py.conf', {
                'project': self.project,
                'version': self.version,
                'master_doc': master_doc,
            })
        conf_file = os.path.join(docs_dir, 'conf.py')
        safe_write(conf_file, conf_template)

    def get_config_params(self):
        """Get configuration parameters to be rendered into the conf file."""
        # TODO this should be handled better in the theme
        conf_py_path = os.path.join(
            os.path.sep,
            os.path.dirname(
                os.path.relpath(
                    self.config_file,
                    self.project.checkout_path(self.version.slug)
                )
            ),
            '',
        )
        remote_version = self.version.commit_name

        github_user, github_repo = version_utils.get_github_username_repo(
            url=self.project.repo)
        github_version_is_editable = (self.version.type == 'branch')
        display_github = github_user is not None

        bitbucket_user, bitbucket_repo = version_utils.get_bitbucket_username_repo(  # noqa
            url=self.project.repo)
        bitbucket_version_is_editable = (self.version.type == 'branch')
        display_bitbucket = bitbucket_user is not None

        gitlab_user, gitlab_repo = version_utils.get_gitlab_username_repo(
            url=self.project.repo)
        gitlab_version_is_editable = (self.version.type == 'branch')
        display_gitlab = gitlab_user is not None

        # Avoid hitting database and API if using Docker build environment
        if getattr(settings, 'DONT_HIT_API', False):
            versions = self.project.active_versions()
            downloads = self.version.get_downloads(pretty=True)
        else:
            versions = self.project.api_versions()
            downloads = api.version(self.version.pk).get()['downloads']

        data = {
            'html_theme': 'sphinx_rtd_theme',
            'html_theme_import': 'sphinx_rtd_theme',
            'current_version': self.version.verbose_name,
            'project': self.project,
            'version': self.version,
            'settings': settings,
            'conf_py_path': conf_py_path,
            'api_host': getattr(
                settings,
                'PUBLIC_API_URL',
                'https://readthedocs.org',
            ),
            'commit': self.project.vcs_repo(self.version.slug).commit,
            'versions': versions,
            'downloads': downloads,

            # GitHub
            'github_user': github_user,
            'github_repo': github_repo,
            'github_version': remote_version,
            'github_version_is_editable': github_version_is_editable,
            'display_github': display_github,

            # BitBucket
            'bitbucket_user': bitbucket_user,
            'bitbucket_repo': bitbucket_repo,
            'bitbucket_version': remote_version,
            'bitbucket_version_is_editable': bitbucket_version_is_editable,
            'display_bitbucket': display_bitbucket,

            # GitLab
            'gitlab_user': gitlab_user,
            'gitlab_repo': gitlab_repo,
            'gitlab_version': remote_version,
            'gitlab_version_is_editable': gitlab_version_is_editable,
            'display_gitlab': display_gitlab,

            # Features
            'dont_overwrite_sphinx_context': self.project.has_feature(
                Feature.DONT_OVERWRITE_SPHINX_CONTEXT
            ),
        }

        finalize_sphinx_context_data.send(
            sender=self.__class__,
            build_env=self.build_env,
            data=data,
        )

        return data

    def append_conf(self, **__):
        """Find or create a ``conf.py`` with a rendered ``doc_builder/conf.py.tmpl`` appended"""
        if self.config_file is None:
            master_doc = self.create_index(extension='rst')
            self._write_config(master_doc=master_doc)

        try:
            self.config_file = (
                self.config_file or
                self.project.conf_file(self.version.slug)
            )
            outfile = codecs.open(self.config_file, encoding='utf-8', mode='a')
        except (ProjectConfigurationError, IOError):
            trace = sys.exc_info()[2]
            six.reraise(
                ProjectConfigurationError,
                ProjectConfigurationError(
                    ProjectConfigurationError.NOT_FOUND
                ),
                trace
            )

        # Append config to project conf file
        tmpl = template_loader.get_template('doc_builder/conf.py.tmpl')
        rendered = tmpl.render(self.get_config_params())

        with outfile:
            outfile.write('\n')
            outfile.write(rendered)

        # Print the contents of conf.py in order to make the rendered
        # configfile visible in the build logs
        self.run(
            'cat',
            os.path.relpath(
                self.config_file,
                self.project.checkout_path(self.version.slug),
            ),
            cwd=self.project.checkout_path(self.version.slug),
        )

    def build(self):
        self.clean()
        project = self.project
        build_command = [
            'python',
            self.python_env.venv_bin(filename='sphinx-build'),
            '-T',
        ]
        if self._force:
            build_command.append('-E')
        if self.config.sphinx.fail_on_warning:
            build_command.append('-W')
        build_command.extend([
            '-b',
            self.sphinx_builder,
            '-d',
            '_build/doctrees-{format}'.format(format=self.sphinx_builder),
            '-D',
            'language={lang}'.format(lang=project.language),
            '.',
            self.sphinx_build_dir,
        ])
        import os
        os.environ['DGLBACKEND'] = 'mxnet'
        cmd_ret_1 = self.run(
            *build_command,
            cwd=os.path.dirname(self.config_file),
            bin_path=self.python_env.venv_bin(),
        )
        os.environ['DGLBACKEND'] = 'pytorch'
        cmd_ret_2 = self.run(
            *build_command,
            cwd=os.path.dirname(self.config_file),
            bin_path=self.python_env.venv_bin(),
        )
        return cmd_ret_1.successful and cmd_ret_2.successful


class HtmlBuilder(BaseSphinx):
    type = 'sphinx'
    sphinx_build_dir = '_build/html'

    def __init__(self, *args, **kwargs):
        super(HtmlBuilder, self).__init__(*args, **kwargs)
        self.sphinx_builder = 'readthedocs'

    def move(self, **__):
        super(HtmlBuilder, self).move()
        # Copy JSON artifacts to its own directory
        # to keep compatibility with the older builder.
        json_path = os.path.abspath(
            os.path.join(self.old_artifact_path, '..', 'json')
        )
        json_path_target = self.project.artifact_path(
            version=self.version.slug, type_='sphinx_search'
        )
        if os.path.exists(json_path):
            if os.path.exists(json_path_target):
                shutil.rmtree(json_path_target)
            log.info('Copying json on the local filesystem')
            shutil.copytree(
                json_path,
                json_path_target
            )
        else:
            log.warning(
                'Not moving json because the build dir is unknown.'
            )


class HtmlDirBuilder(HtmlBuilder):
    type = 'sphinx_htmldir'

    def __init__(self, *args, **kwargs):
        super(HtmlDirBuilder, self).__init__(*args, **kwargs)
        self.sphinx_builder = 'readthedocsdirhtml'


class SingleHtmlBuilder(HtmlBuilder):
    type = 'sphinx_singlehtml'

    def __init__(self, *args, **kwargs):
        super(SingleHtmlBuilder, self).__init__(*args, **kwargs)
        self.sphinx_builder = 'readthedocssinglehtml'


class LocalMediaBuilder(BaseSphinx):
    type = 'sphinx_localmedia'
    sphinx_builder = 'readthedocssinglehtmllocalmedia'
    sphinx_build_dir = '_build/localmedia'

    @restoring_chdir
    def move(self, **__):
        log.info('Creating zip file from %s', self.old_artifact_path)
        target_file = os.path.join(
            self.target,
            '{}.zip'.format(self.project.slug),
        )
        if not os.path.exists(self.target):
            os.makedirs(self.target)
        if os.path.exists(target_file):
            os.remove(target_file)

        # Create a <slug>.zip file
        os.chdir(self.old_artifact_path)
        archive = zipfile.ZipFile(target_file, 'w')
        for root, __, files in os.walk('.'):
            for fname in files:
                to_write = os.path.join(root, fname)
                archive.write(
                    filename=to_write,
                    arcname=os.path.join(
                        '{}-{}'.format(self.project.slug, self.version.slug),
                        to_write),
                )
        archive.close()


class EpubBuilder(BaseSphinx):
    type = 'sphinx_epub'
    sphinx_builder = 'epub'
    sphinx_build_dir = '_build/epub'

    def move(self, **__):
        from_globs = glob(os.path.join(self.old_artifact_path, '*.epub'))
        if not os.path.exists(self.target):
            os.makedirs(self.target)
        if from_globs:
            from_file = from_globs[0]
            to_file = os.path.join(
                self.target,
                '{}.epub'.format(self.project.slug),
            )
            self.run(
                'mv',
                '-f',
                from_file,
                to_file,
                cwd=self.project.checkout_path(self.version.slug),
            )


class LatexBuildCommand(BuildCommand):

    """Ignore LaTeX exit code if there was file output."""

    def run(self):
        super(LatexBuildCommand, self).run()
        # Force LaTeX exit code to be a little more optimistic. If LaTeX
        # reports an output file, let's just assume we're fine.
        if PDF_RE.search(self.output):
            self.exit_code = 0


class DockerLatexBuildCommand(DockerBuildCommand):

    """Ignore LaTeX exit code if there was file output."""

    def run(self):
        super(DockerLatexBuildCommand, self).run()
        # Force LaTeX exit code to be a little more optimistic. If LaTeX
        # reports an output file, let's just assume we're fine.
        if PDF_RE.search(self.output):
            self.exit_code = 0


class PdfBuilder(BaseSphinx):

    """Builder to generate PDF documentation."""

    type = 'sphinx_pdf'
    sphinx_build_dir = '_build/latex'
    pdf_file_name = None

    def build(self):
        self.clean()
        cwd = os.path.dirname(self.config_file)

        # Default to this so we can return it always.
        self.run(
            'python',
            self.python_env.venv_bin(filename='sphinx-build'),
            '-b',
            'latex',
            '-D',
            'language={lang}'.format(lang=self.project.language),
            '-d',
            '_build/doctrees',
            '.',
            '_build/latex',
            cwd=cwd,
            bin_path=self.python_env.venv_bin(),
        )
        latex_cwd = os.path.join(cwd, '_build', 'latex')
        tex_files = glob(os.path.join(latex_cwd, '*.tex'))

        if not tex_files:
            raise BuildEnvironmentError('No TeX files were found')

        # Run LaTeX -> PDF conversions
        pdflatex_cmds = [
            ['pdflatex', '-interaction=nonstopmode', tex_file]
            for tex_file in tex_files]  # yapf: disable
        makeindex_cmds = [
            ['makeindex', '-s', 'python.ist', '{0}.idx'.format(
                os.path.splitext(os.path.relpath(tex_file, latex_cwd))[0])]
            for tex_file in tex_files]  # yapf: disable

        if self.build_env.command_class == DockerBuildCommand:
            latex_class = DockerLatexBuildCommand
        else:
            latex_class = LatexBuildCommand
        pdf_commands = []
        for cmd in pdflatex_cmds:
            cmd_ret = self.build_env.run_command_class(
                cls=latex_class, cmd=cmd, cwd=latex_cwd, warn_only=True)
            pdf_commands.append(cmd_ret)
        for cmd in makeindex_cmds:
            cmd_ret = self.build_env.run_command_class(
                cls=latex_class, cmd=cmd, cwd=latex_cwd, warn_only=True)
            pdf_commands.append(cmd_ret)
        for cmd in pdflatex_cmds:
            cmd_ret = self.build_env.run_command_class(
                cls=latex_class, cmd=cmd, cwd=latex_cwd, warn_only=True)
            pdf_match = PDF_RE.search(cmd_ret.output)
            if pdf_match:
                self.pdf_file_name = pdf_match.group(1).strip()
            pdf_commands.append(cmd_ret)
        return all(cmd.successful for cmd in pdf_commands)

    def move(self, **__):
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        exact = os.path.join(
            self.old_artifact_path,
            '{}.pdf'.format(self.project.slug),
        )
        exact_upper = os.path.join(
            self.old_artifact_path,
            '{}.pdf'.format(self.project.slug.capitalize()),
        )

        if self.pdf_file_name and os.path.exists(self.pdf_file_name):
            from_file = self.pdf_file_name
        if os.path.exists(exact):
            from_file = exact
        elif os.path.exists(exact_upper):
            from_file = exact_upper
        else:
            from_globs = glob(os.path.join(self.old_artifact_path, '*.pdf'))
            if from_globs:
                from_file = max(from_globs, key=os.path.getmtime)
            else:
                from_file = None
        if from_file:
            to_file = os.path.join(
                self.target, '{}.pdf'.format(self.project.slug))
            self.run(
                'mv',
                '-f',
                from_file,
                to_file,
                cwd=self.project.checkout_path(self.version.slug),
            )
