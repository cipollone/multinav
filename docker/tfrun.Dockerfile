FROM cipollone/multinav:develop

COPY . ./

ENTRYPOINT ["bash", "-l", "docker/tfrun-entry.sh"]
